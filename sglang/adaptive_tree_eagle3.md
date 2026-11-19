# Load-adaptive tree drafting for EAGLE-3 in SGLang

Status: 2026-11-19. One commit on upstream SGLang base `31e55a5120` (local sglang branch `SGLang-adaptive-tree`,
commit `8f591b9147`). Apply with `git apply` on that base. Full diff: [`SGLang-adaptive-tree.diff`](SGLang-adaptive-tree.diff)
(11 files, +170 / −29).

Setup for every number here:
- **Model:** Llama-3.1-8B-Instruct with the lmsys EAGLE-3 head.
- **Hardware:** 1× RTX 6000 Ada (48 GB).
- **Task:** GSM8K prompts at T=0.6, top-p 0.9. "Throughput" is total output tokens/s.
- **Repeats:** where noted, the mean of 3 repeats.

---

## 1. What we changed and why it is faster

### The problem

EAGLE-3 runs with one fixed draft shape, chosen at server launch. At concurrency 1–8 a large draft
tree is best: it accepts about 4.7 tokens per step, against about 3.1 for a chain. From concurrency 16
on, a short chain is best. With a fixed shape, either choice gives up a lot at the other end:

| one fixed shape for all loads | what goes wrong |
|---|---|
| tree (N = 32–60 verified tokens) | up to 78% slower than the best shape at concurrency 64 |
| short chain | 25–30% slower than the best shape at concurrency 1–4 |

The cause is a **compute budget**. One target forward pass costs about the same for up to about 110
tokens on this GPU, because it is memory-bound. Above that, every extra token costs about 0.17 ms.
A step verifies concurrency × N tokens:
- **Under the budget,** extra tree nodes are free, so the tree's higher acceptance is pure gain.
- **Over the budget,** a tree keeps only about 6% of the nodes it verifies, and the extra compute is
  wasted.

SGLang already has **adaptive speculation** (`--speculative-adaptive`). It pre-builds one runtime
state per draft depth (CUDA graphs plus attention buffers) and switches between them by batch size
and measured acceptance. But it supported **chains only**: it rejected `topk != 1` and hard-wired
N = steps + 1. So it could never use the tree's low-load advantage. Its default policy also turns
speculation off at batch 64, where a short chain was still about 50% faster than plain decoding.

### What we built

1. **Adaptive states can be trees.** Each adaptive state is now a tree shape (depth, N, k) instead of
   a chain depth. Two maps in the existing adaptive config, `draft_tokens` (steps → N) and
   `draft_topk` (steps → k), give each state its own pruned EAGLE tree. Each state gets its own
   CUDA graphs and attention backends, so a low-load state can be a wide tree and a high-load state a
   true k=1 chain. Configs without the new keys behave exactly as before.
2. **Switching k between steps is safe.** After each verify, the draft-extend pass computes the
   root's top-k candidates for the next step. It now always computes them at the server's largest k,
   and each state slices the prefix it needs. `torch.topk` output is sorted, so the slice is exactly
   that state's top-k, and switching works in both directions (tree ↔ chain).
3. **A budget-derived policy.** Each state is used only where concurrency × N stays within the free
   budget of about 110 tokens. This is `hsd_eval/configs/adaptive_tree_budget.json`:

   | running requests | state | max tokens verified per pass |
   |---|---|---|
   | 1–3 | tree k=4, depth 6, N=32 | 96 |
   | 4–7 | tree k=4, depth 5, N=16 | 112 |
   | 8–15 | tree k=4, depth 4, N=7 | 105 |
   | 16+ | chain k=1, depth 3, N=4 | 64–256 |

4. **No CPU stalls in tree steps.** Profiling showed the GPU idle for 2.2–2.8 ms per tree step. The
   cause was two blocking GPU→CPU reads during attention planning. SGLang has sync-free versions of
   both, but they were opt-in, and the verify one was restricted to chains. They now default on for
   trees (topk > 1), and an explicit environment setting still wins.

   | tree N=32 | before | after |
   |---|---|---|
   | GPU idle per step, concurrency 1 | 2.2 ms | 0.8 ms |
   | step time, concurrency 1 | 25.8 ms | 24.4 ms (−5.5%) |
   | step time, concurrency 4 | 31.4 ms | 30.7 ms (−2.2%) |

   Greedy outputs are identical to before.
5. **Measurement.** A server counter for tokens actually verified (batch size × active N) is exposed
   in `/server_info` as `spec_verify_token_ct` and `spec_verify_pass_ct`. SGLang's existing
   per-request `spec_accept_rate` assumes the startup N, so it is wrong once states switch.

### Results (mean of 3 repeats, total tok/s)

**Ours vs SGLang's existing adaptive mode (chains):**

| context | mode | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---|---|---|---|---|---|---|---|
| ~115 prompt tokens | SGLang adaptive (chains) | 143 | 255 | 423 | 700 | 998 | 1525 | 1807 |
| ~115 prompt tokens | **ours (tree-aware, budget)** | **183** | **319** | **525** | **756** | **1128** | **1695** | 1790 |
| ~115 prompt tokens | ours vs SGLang | +28% | +25% | +24% | +8% | +13% | +11% | −1% |
| ~1.85k prompt tokens | SGLang adaptive (chains) | 138 | 263 | 469 | 669 | 880 | 1599 | 1795 |
| ~1.85k prompt tokens | **ours (tree-aware, budget)** | **186** | **340** | **601** | **718** | **1098** | **1833** | **1972** |
| ~1.85k prompt tokens | ours vs SGLang | +35% | +29% | +28% | +7% | +25% | +15% | +10% |

**Ours vs fixed trees:**

| | short context | long context |
|---|---|---|
| vs the best fixed tree at each point (chosen in hindsight) | +1% to +15%, except −10% at c=4–8 | +2% to +40%, except −17% at c=8 |
| vs the best single fixed tree for all loads (N7), mean | **+7%** | **+15%** |
| vs large fixed trees (N32 / N60), mean | +61% / +159% | +71% / +173% |

Against the per-point best fixed config of either structure (the oracle), ours is −3% on average at
short context and +1% at long context. SGLang's adaptive mode is −16% at both.

Tokens verified per pass:

| | c=1 | c=4 | c=16 | c=64 |
|---|---|---|---|---|
| ours | 32 | 62 | 67 | 137 |
| SGLang adaptive | 4 | 14 | 35 | 57 |

Ours fills the free budget at low load. SGLang's adaptive mode leaves it unused.

### Accuracy (task accuracy unchanged; an open first-token distribution shift in adaptive mode)

Verification is unchanged: stock SGLang target-only tree sampling, which is lossless for any draft
tree. Our changes only select and draft the tree. The risk is a state-switching bug corrupting
verification inputs. Checks so far, on GSM8K with sgl-eval (1319 questions, T=0.6):

| check | target-only | ours | SGLang adaptive |
|---|---|---|---|
| GSM8K at concurrency 8 | 82.79% | 81.88% | 81.96% |
| GSM8K at concurrency 64 | 81.65% | 82.26% | 81.96% |
| greedy at concurrency 8, identical to target-only | – | 30/64 | 38/64 |
| largest top-2 log-prob gap at a difference | – | 0.125 | 0.250 |
| distribution test under switching (Bonferroni p, 18 tests) | – | 0.030 | 0.296 |

- **GSM8K:** within 1 standard error (±1.06) of target-only.
- **Greedy:** every difference starts at a bf16 near-tie, a numerics effect of batching that SGLang's
  own adaptive mode shows too.
- **Distribution test:** inconclusive. A single statistic has raw p = 0.002.
- **Still to come:**
  - a 20,000-sample rerun of the distribution test
  - a lossy positive control
  - a temperature sweep (GSM8K at T = 0 / 0.3 / 1.0, distribution test at T = 0.5)

### Known limitations and next steps

- **States are keyed by depth.** Two states cannot share a depth, so the mid-load states use depths 5
  and 4. The best fixed trees at c=4–8 use depth 6, which is where ours is 10–17% behind. Next step:
  key states by (steps, N, k).
- **The slots are set by hand.** They come from the measured budget. A cost model calibrated online
  would set them automatically per GPU and context length.
- **Memory:** each extra state costs about 1.7–2.4 GB of CUDA graphs, about 6.2 GB for 4 states.
- **Noise:** repeat spread at c ≥ 8 is ±10–20%. Differences under about 10% there are not conclusive.

---

## 2. Diff: stock EAGLE-3 in SGLang vs this branch

How each component behaves before and after:

| component | stock SGLang (base `31e55a5120`) | this branch |
|---|---|---|
| Adaptive mode and trees | `--speculative-adaptive` disabled itself when `--speculative-eagle-topk != 1` | trees allowed; each state has its own k |
| Draft tokens per state | always `steps + 1` (a chain) | `draft_tokens` map, default `steps + 1` |
| Draft width per state | one server-wide `topk` | `draft_topk` map, default server topk |
| Carried root candidates | width = active topk | width = startup topk (the maximum); each state slices |
| Buffer sizing | `max(candidate steps) + 1` | max over configured N |
| Sync-free verify plan | opt-in, chains only | default on for trees; the env var overrides |
| Sync-free draft indptr | opt-in | default on for trees; the env var overrides |
| Verified-token counter | none | `spec_verify_token_ct`, `spec_verify_pass_ct` in `/server_info` |

### Files

| file | lines | change |
|---|---|---|
| `srt/speculative/adaptive_spec_params.py` | +36 / −5 | drop the topk=1 rejection; parse and validate `draft_tokens` / `draft_topk`; `draft_tokens_for_step`, `draft_topk_for_step` |
| `srt/speculative/adaptive_runtime_state.py` | +37 / −1 | `SpecRuntimeState.speculative_eagle_topk`; build each state with its N and k; pin the startup state's k |
| `srt/speculative/eagle_worker_v2.py` | +56 / −8 | `carry_topk`; slice carried candidates in `draft()`; draft extend at carry width; build/override/apply topk per state |
| `srt/layers/attention/flashinfer_backend.py` | +12 / −3 | `_host_plan_enabled`: sync-free verify and draft planning default on for topk > 1 |
| `srt/managers/scheduler_components/metrics_reporter.py`, `internal_state.py` | +13 | verified-token and verify-pass counters |
| `srt/speculative/spec_info.py`, `model_executor/model_runner.py`, `arg_groups/speculative_hook.py` | +9 / −12 | size buffers and resolve startup N from the configured map |
| `srt/environ.py` | +1 | comment documenting the tree default of the two host-plan flags |
| `test/registered/unit/spec/test_adaptive_runtime_state.py` | +6 | test fake implements the two new policy methods |

### Key hunks

**Trees allowed in adaptive mode** (`adaptive_spec_params.py`):

```diff
-    if cfg.speculative_eagle_topk is not None and cfg.speculative_eagle_topk != 1:
-        return (
-            f"speculative_eagle_topk={cfg.speculative_eagle_topk} "
-            "(only topk=1 is supported)"
-        )
```

**Each state built at its own size and width** (`adaptive_runtime_state.py`):

```diff
             state = self.worker.build_adaptive_runtime_state(
                 speculative_num_steps=steps,
-                speculative_num_draft_tokens=steps + 1,
+                speculative_num_draft_tokens=self.params.draft_tokens_for_step(steps),
                 cuda_graph_bs=pruned_bs,
+                **({} if topk is None else {"speculative_eagle_topk": topk}),
             )
```

**Carry the widest top-k; each state slices it** (`eagle_worker_v2.py`):

```diff
+        self.carry_topk = self.topk          # startup topk, the widest any state may use
 ...
     def draft(self, batch, ...):
         draft_input: EagleDraftInput = batch.spec_info
+        if self.topk < self.carry_topk and draft_input.topk_p is not None:
+            draft_input.topk_p = draft_input.topk_p[:, : self.topk]
+            draft_input.topk_index = draft_input.topk_index[:, : self.topk]
 ...
-            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
+            topk_p, topk_index = fast_topk(probs, self.carry_topk, dim=-1)
```

**Switching sets k with depth and N** (`eagle_worker_v2.apply_runtime_state`; the graph-capture
override is the same):

```diff
         self.speculative_num_steps = state.speculative_num_steps
         self.speculative_num_draft_tokens = state.speculative_num_draft_tokens
+        if state.speculative_eagle_topk is not None:
+            self.topk = state.speculative_eagle_topk
 ...
+        if state.speculative_eagle_topk is not None:
+            dw.topk = state.speculative_eagle_topk
```

**Sync-free planning for trees** (`flashinfer_backend.py`):

```diff
+def _host_plan_enabled(flag, topk: int) -> bool:
+    return flag.get() if flag.is_set() else topk > 1
 ...
-            and envs.SGLANG_OPT_FLASHINFER_VERIFY_HOST_PLAN.get()
             and spec_info.spec_input_type == SpecInputType.EAGLE_VERIFY
-            and spec_info.topk == 1
+            and _host_plan_enabled(
+                envs.SGLANG_OPT_FLASHINFER_VERIFY_HOST_PLAN, spec_info.topk
+            )
 ...
-            envs.SGLANG_OPT_FLASHINFER_DRAFT_HOST_INDPTR.get()
+            _host_plan_enabled(envs.SGLANG_OPT_FLASHINFER_DRAFT_HOST_INDPTR, self.topk)
```

### Tried and reverted

Both are measured in `handover.md` §9.

- **Verify attention without split-KV:** −19% at long context, and not bit-exact.
- **Reusing a zeroed scratch buffer in verify sampling:** exact, but no measurable gain (about 0.3%
  of a step).

### Running it

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --speculative-adaptive --speculative-adaptive-config hsd_eval/configs/adaptive_tree_budget.json \
  --speculative-eagle-topk 4 --speculative-num-steps 6 --dtype bfloat16
```

`--speculative-eagle-topk` must be the widest k of any state. `--speculative-num-steps` selects the
starting state, and that state must use the startup k.

### Reproducing the numbers (`hsd_eval/`)

| script | what it does |
|---|---|
| `run_oracle_reps.sh` + `analyze_oracle_reps.py` | 3-repeat grid: all fixed configs plus both adaptive modes |
| `roofline_check.py`, `hw_peaks.py` | ideal vs measured step time |
| `profile_points.sh` + `trace_breakdown.py` | profiler traces and per-step GPU busy/idle breakdown |
| `ab_test.sh` | A/B of a server flag: greedy exactness plus speed |
| `accuracy_check.sh`, `temp_check.sh`, `dist_followup.sh` + `analyze_accuracy.py` | losslessness checks |
