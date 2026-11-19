# Handover: HSD (Hierarchical Speculative Decoding) in SGLang

Status as of 2026-11-18. Goal: an upstream PR to sgl-project/sglang adding tree HSD
([arXiv 2601.05724](https://arxiv.org/abs/2601.05724),
[reference code](https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding)).

Summary:
- **Accuracy: done for tree HSD.** It is lossless by every check we ran. The separate adaptive-tree branch has an open first-token distribution shift in adaptive mode (Open items, item 10).
- **Performance: measured, not optimized.**
  - At concurrency 1, tree HSD accepts about 1.3 more tokens per verify step than chain HSD. It is only about 4% faster, because our tree mode runs the draft without CUDA graphs.
  - At high concurrency every tree loses to a chain, and stock SGLang EAGLE-3 trees do too. This is a compute-budget effect, not a bug.
- **Main next steps:**
  - draft CUDA graphs in tree mode
  - pruned (budgeted) sampled trees
  - choosing the draft shape by batch size

Hardware note: the GPUs on this box are **NVIDIA RTX 6000 Ada Generation** (48 GB, Ada), not the
RTX A6000 (Ampere). `nvidia-smi` shows the name. All speed numbers below are for the Ada card.

## Where things are

All paths are relative to `~/project/results_analysis/spd/sglang_spd/`.

| Path | What | Branch |
|---|---|---|
| `sglang/` | SGLang main (base `31e55a5120`) + our changes, committed as `d7a1b739b3` | `SGLang-tree-hsd` |
| `Hierarchical-Speculative-Decoding/` | HSD reference repo, unchanged | `sglang-tree-hsd` |
| `hsd_eval/` | Evaluation harness and results (`runs/`); see its README.md | n/a |

Committed locally, not pushed: `git -C sglang show --stat SGLang-tree-hsd` lists the 12
files. An empty `tree-hsd` branch (same base, no commits) also exists.

## What was implemented

**Single-draft HSD.** Lossless single-draft HSD is the same rule as block verification
(the HSD README states this, and our kernel test confirms it decision for decision).
SGLang already has block verification, so `--speculative-use-hsd` with `topk=1` just turns
on `--speculative-use-block-verification`.

**Tree HSD (the new part).** It has three pieces:

1. **Sampled draft tree.** With `topk > 1` under rejection sampling, each expanded EAGLE
   node now draws its `topk` children from q *without replacement*, using Gumbel top-k
   (children are in draw order), and keeps q for every expanded node. Before this change,
   SGLang's rejection sampling asserted `topk == 1`.
2. **Verifier.** `python/sglang/kernels/ops/speculative/sampled_tree_verify.py` is a Triton
   kernel with one program per request. It walks the tree like `hsd()` in the reference
   `tree-hsd/sampled_tree.py`: it block-verifies the first-child path from a child; a stop
   at depth j continues at that node with the residual, and a stop at 0 tries the next
   sibling under `q` with the tried siblings removed. The same kernel with 1-node paths is
   RRSw (recursive rejection sampling without replacement), the baseline.
3. **Flags.**
   - `--speculative-use-rejection-sampling` with `topk > 1`: sampled tree, verified by RRSw.
   - `--speculative-use-hsd` with `topk > 1`: sampled tree, verified by tree HSD.
   - The tree must not be pruned: `--speculative-num-draft-tokens = 1 + k + (steps-1)*k^2`
     (e.g. k=4, steps=6 gives 85). Pruning a drawn child by its score biases the remaining
     draws, so the arg hook rejects other values.

Layout contract the kernel relies on: node n's children were drawn from `draft_probs` row
`(retrieve_next_token[n] - 1) // topk`, and siblings are linked in draw order. EAGLE's
`organize_draft_results` / `build_tree_kernel_efficient` produce exactly this when nothing is
pruned. The tests build trees with those functions so the contract stays checked.

### Files touched in `sglang/`

- `kernels/ops/speculative/sampled_tree_verify.py` (new): the verifier.
- `kernels/ops/speculative/verify_sampling.py`: exports it.
- `srt/speculative/eagle_utils.py`: routes verify to it for `topk > 1`, plus its coin shape.
- `srt/speculative/eagle_worker_v2.py`: samples tree children, stacks per-node q, skips
  draft CUDA graphs in tree mode.
- `srt/speculative/spec_utils.py`: `sample_draft_proposal(num_samples=...)`.
- `srt/arg_groups/{fields/spec.py, field_order.py, speculative_hook.py}`: the new flag
  and its validation.
- Tests: the new `test/registered/kernels/ops/speculative/test_sampled_tree_verify.py`, and
  updates to three existing unit tests (new fields in fakes; topk>1 is no longer an error).

## Accuracy results so far

### Kernel level (exact, synthetic distributions)

`test_sampled_tree_verify.py` has 8 cases, all passing, stable over 6 repeated runs:
- With `topk=1`, the tree kernel matches SGLang's existing block-verification kernel
  decision for decision under the same coins.
- Losslessness, for both tree HSD and RRSw: on trees built by SGLang's own tree code,
  the first 3 committed tokens follow the exact target joint (chi-square). This includes
  zero-probability tokens, so nodes can have fewer than `topk` real children.
- Power checks: the harness flags top-k (non-sampled) trees as biased, and HSD accepts at
  least as many tokens as RRSw on the same trees.
- Mutation check (done by hand, not committed): breaking the sibling removal or inflating
  the whole-path acceptance makes the losslessness tests fail.

Existing tests: `test/registered/unit/spec` + `unit/speculative` pass (506). The one failure
there, `test_dflash_model_config.py::test_gemma_attention...`, also fails on pristine HEAD.
`test/registered/kernels/ops/speculative` passes (373); `test_dspark_npu_dispatch.py`
cannot be collected without NPU, also on HEAD.

### End to end (Llama-3.1-8B-Instruct + lmsys EAGLE-3 head, 1x RTX 6000 Ada)

GSM8K uses sgl-eval 0.1.3 on all 1319 questions, T=0.6, top-p 0.9. The lossless column
compares the first 8 tokens of 3000 T=1 samples per prompt against target-only (2 prompts,
18 chi-square tests, Bonferroni).

| Config | GSM8K | Tokens/step | Lossless p |
|---|---|---|---|
| target-only | 82.79% | n/a | reference |
| chain, token-wise RS (5 steps) | 82.18% | 3.340 | 1.0 |
| chain, HSD (= block verification) | 81.73% | 3.371 | 1.0 |
| tree RRSw (k=4, 6 steps, 85 nodes) | 81.05% | 4.752 | 0.94 |
| tree HSD (same tree) | 80.29% | 4.751 | 0.59 |
| lossy positive control (accept thresholds 0.3) | 79.23% | 5.300 | **0** |

The positive control is caught (p = 0), so the distribution test has power at 3000 samples.
One GSM8K standard error is about 1.05 points. Tree HSD's 2.5-point gap (about 1.7σ) was
followed up with 20,000 samples of the prompt where its p-values ran low: there is no
deviation (Bonferroni p = 0.22 over 10 tests, including the joint of the first 3 tokens).
Reproduce with `cd hsd_eval && source env.sh && python analyze.py runs`; the 20k run is
`big_dist.sh` → `runs/big_prompt1/`.

### Targeted checks on 5 fixed prompts

**(1) Same inputs at T=0 and T=0.01, against target-only greedy** (`check_greedy.py`,
`compare_greedy.py`, results in `runs/greedy/`). T=0 uses SGLang's greedy tree check (the
new kernel is not called); T=0.01 sends the batch through the sampled-tree / HSD kernel.
- Tree HSD at T=0.01 matches greedy on 5 of 5 prompts.
- Every mismatch in any config is at a position where the target's top-2 log-probs differ
  by 0, 0.0625 or 0.125, i.e. one bf16 step: a numerical tie. Target-only itself diverges at
  two of these. No divergence happens at a clear gap.

**(2) Exact losslessness on real distributions** (`exact_real.py`, results in
`runs/exact/`). For each prompt, real target `p` and EAGLE-3 `q` are captured at every
context of a small full tree, restricted to the context's top-m tokens (m=4 with depth 2,
and m=3 with depth 3; 2 children per node). The paper's naive Python verifiers
(`hsd_tree_wor`, `rrsw_tree` in `tree-hsd/verify_sampled_tree_wor.py`) are enumerated over
every sampled tree and walk outcome.
- Output law vs target joint: total variation at most 8.6e-14 (float round-off) for both
  HSD and RRSw. Lossless.
- Contrast: the same HSD on top-k (unsampled) trees is off by TV 0.08 to 0.57.
- Our Triton kernel vs the naive per-tree law: 60 comparisons (3 likeliest trees × HSD/RRSw
  × 2 shapes × 5 prompts, 200k coin draws each). Max TV 2.6e-3, which is the Monte Carlo
  noise level. Chi-square p-values are uniform (KS p = 0.85); the smallest, Bonferroni-corrected,
  is 0.42.
- HSD accepts at least as many tokens per step as RRSw in all 10 cases (+0 to +0.012).

Capturing `q(.|c)` needs care: after a prefill, EAGLE drafts the token *after* the target's
sampled token. So `exact_real.py` prefills `c[:-1]` and forces the target to emit `c[-1]`
with `logit_bias`. The dump hook is a debug-only patch in the worktree
`/data/twang/sglang_debug` (not committed); `run_exact.sh` uses it.

Expectation: on this plain score-selected tree, the paper measures tree HSD only about
+0.5% over RRSw. Its larger gains (+1.5% to +3.3%) come from "hybrid/spine" trees, which
are not implemented here (see open items).

## Performance results (2026-11-17 / 18)

Same model pair and GPU as above. GSM8K through sgl-eval, T=0.6, top-p 0.9, max 1024 tokens.
Glossary:
- **k**: children per node (`--speculative-eagle-topk`).
- **d**: draft depth (`--speculative-num-steps`).
- **nodes / N**: tokens the target verifies per request per step (`--speculative-num-draft-tokens`).
  - Our sampled trees are unpruned, so N = 1 + k + (d−1)k². A chain has N = d + 1.
  - Stock EAGLE trees prune to any N.
- **accept**: tokens committed per verify step, including the bonus token. This is the server's own
  `avg_spec_accept_length`.
- **per-request speed**: Σ tokens ÷ Σ per-answer latency, the speed one user sees. It already
  normalizes for answer length.
- **total throughput**: all tokens ÷ wall time, what the whole server produces.

At concurrency 1 the two speeds are the same number.

### 1. Full GSM8K (1319 questions), first configs

The chains here draft **5** tokens (steps 5, 6 draft tokens) and the trees draft **6** (k=4,
steps 6, 85 nodes). That mismatch was found later, so compare chain vs tree with section 2.
The trees ran with `--max-running-requests 32 --mem-fraction-static 0.6`.

| config | accept c=64 / c=1 | acc c=64 / c=1 | per-request tok/s c=64 / c=1 | total tok/s c=64 |
|---|---|---|---|---|
| target-only | – | 82.79% / 82.49% | 36.0 / 58.9 | 1995 |
| chain, token-wise | 3.34 / 3.32 | 82.18% / 81.50% | 28.3 / 143.5 | 1774 |
| chain, HSD | 3.37 / 3.36 | 81.73% / 81.58% | 31.1 / 144.3 | 1890 |
| tree, RRSw (token-wise) | 4.75 / 4.71 | 81.05% / 80.89% | 5.2 / 152.3 | 326 |
| tree, HSD | 4.75 / 4.74 | 80.29% / 81.58% | 4.8 / 148.2 | 303 |
| stock EAGLE-3 tree (k10 d5 N60) | – / 4.65 | – / 81.73% | – / 177.1 | – |
| lossy control (thresholds 0.3) | 5.30 / – | 79.23% / – | 5.9 / – | 370 |

- **c=64:** the c=64 runs are `run_configs.sh` with sgl-eval's default concurrency of 64; results in
  `runs/<config>/`.
- **c=1:** the c=1 runs are `run_gsm8k_bs1.sh`; results in `runs/gsm8k_bs1/`.
- **Accuracy:** GSM8K accuracy is within noise everywhere (1 SE ≈ 1.05 points). Tree HSD reached 81.58% at c=1,
  so its 80.29% at c=64 was noise.

### 2. Matched settings, first 100 questions (`hsd_eval/run_hsd100.sh`, results in `runs/hsd100/`)

Matched settings for all four rows:
- every row drafts **6** tokens deep (chain: steps 6, N=7; tree: k=4, steps 6, N=85)
- same server flags: `--mem-fraction-static 0.6 --max-running-requests 32`
- same sampling and the same grader

| config | SGLang c=1: accept / acc / tok/s | SGLang c=64: accept / acc / per-req / total | reference code, c=1: accept / acc / tok/s |
|---|---|---|---|
| chain, token-wise | 3.45 / 78% / 142.7 | 3.49 / 78% / 34.1 / 1334 | 3.07 / 85% / 96.8 |
| chain, HSD | 3.55 / 85% / 146.9 | 3.58 / 76% / 34.2 / 1194 | 3.22 / 79% / 101.5 |
| tree, RRSw | 4.68 / 87% / 147.3 | 4.75 / 80% / 5.9 / 298 | 4.54 / 84% / 114.5 |
| tree, HSD | 4.86 / 82% / 152.6 | 4.70 / 83% / 6.0 / 292 | 4.66 / 79% / 116.4 |

- **The tree's acceptance lead is real.** It is about +1.2 to 1.4 tok/step over the chain at the same depth, in both
  SGLang and the paper's code. It comes from sibling retries, not the extra depth.
- **HSD over token-wise is a few percent:** about +0.1 to 0.2 tok/step and +2 to 4% speed at c=1. The
  gain should grow with spine/hybrid trees (see open items).
- **Accuracy at n=100** has a standard error of about ±4 points. The 76 to 87% spread is noise.
- **The reference code is slower and accepts less.** It is the paper's own PyTorch code (`hsd_eval/ref_gsm8k.py`) in fp16, and it
  samples the draft at T=1 while the target runs at T=0.6. That is still lossless, but acceptance is lower.
- **c=64 latency includes queueing.** At most 32 requests run at once, so half of them wait in a queue.

The latency view (mean / p90 seconds per answer), for completeness:
- **c=1:** all four configs ≈ 1.17 to 1.19 s / 1.7 s. Tree HSD writes 10 to 15 more tokens in the same time,
  so time per output token is the fairer metric.
- **c=64:** chains ≈ 5.2 s / 7.0 to 7.6 s; trees ≈ 27 s / 38 s.

### 3. Why trees lose at high concurrency: the compute budget

Measured one Llama-3.1-8B bf16 forward pass with N new tokens on this GPU (HF eager,
`fwd_cost.py`, copied to `hsd_eval/fwd_cost.py`):

| N | 1 | 7 | 32 | 85 | 128 | 170 | 224 | 512 | 1024 | 2720 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms | 19.0 | 20.3 | 22.2 | 23.0 | 24.4 | 29.6 | 36.0 | 75.1 | 151.7 | 437.9 |

- **Below about 110 tokens per pass, the forward is memory-bound.** The cost is reading 16 GB of weights, so extra tokens are
  nearly free.
- **Above that it is compute-bound,** at about 0.17 ms per extra token (about 95 TFLOPS effective).
- **Batching and speculation draw on the same free budget.** The limit is:

  **running requests × N (tokens verified per request) ≲ 110 on this GPU**

  - A chain (N=7) stays inside the budget up to about 15 requests.
  - Our 85-node tree is already at the edge with 1 request.
- **Only about 6% of a big tree's nodes get accepted.** The server log's "accept rate" is 0.04 to 0.05 for the tree vs 0.47
  for the chain. Once the forward is compute-bound, every wasted node costs real time.
- **The model matches the measurements:**
  - It predicts the tree's total throughput stays flat at about 350 tok/s at any concurrency; measured ≈ 300.
  - It predicts a batching gain of about 2× for the tree vs about 9× for the chain; measured 2.0× vs 8 to 9×.

### 4. Draft-shape sweep with HSD (`hsd_eval/run_opt.sh`, results in `runs/opt/`)

One fresh server per config, all HSD, `--mem-fraction-static 0.6`, max-running 64 (32 for N ≥ 49).
Questions per concurrency: c=1 and c=4 on 100, c=16 on 200, c=64 on 400.
Per-request / total tok/s:

| config | N | accept | c=1 | c=4 | c=16 | c=64 |
|---|---|---|---|---|---|---|
| target-only | 1 | 1.00 | 58.9 | 56.3 / 222 | 51.2 / 721 | 38.4 / 1732 |
| chain d2 | 3 | 2.37 | 117.2 | 110.2 / 436 | 89.1 / 1146 | 44.5 / 2207 |
| chain d3 | 4 | 2.79 | 132.5 | 122.5 / 484 | 98.8 / 1298 | **45.4 / 2434** |
| chain d4 | 5 | 3.09 | 137.5 | 127.8 / 483 | 91.4 / 1291 | 37.1 / 2186 |
| chain d6 | 7 | 3.51 | 145.2 | **129.3 / 514** | **98.4 / 1360** | 35.7 / 2040 |
| chain d8 | 9 | 3.72 | 139.0 | 125.3 / 493 | 74.2 / 1022 | 27.2 / 1639 |
| tree k2 d3 | 11 | 3.19 | 134.7 | 114.4 / 451 | 71.0 / 963 | 22.9 / 1350 |
| tree k2 d4 | 15 | 3.65 | 146.9 | 124.3 / 491 | 58.6 / 842 | 18.1 / 1037 |
| tree k2 d6 | 23 | 4.23 | 150.4 | 124.9 / 493 | 52.1 / 768 | 16.1 / 946 |
| tree k3 d4 | 31 | 3.85 | 147.7 | 107.1 / 424 | 35.9 / 528 | 10.2 / 615 |
| tree k3 d6 | 49 | 4.48 | **151.1** | 85.8 / 340 | 28.9 / 438 | 8.3 / 499 |
| tree k4 d3 | 37 | 3.43 | 134.5 | 88.0 / 349 | 29.8 / 438 | 8.9 / 523 |
| tree k4 d6 | 85 | 4.66 | 146.9 | 56.2 / 223 | 17.8 / 267 | 5.2 / 306 |

- **The best shape depends on concurrency:**
  - c=1: tree k3 d6 (151); k2 d6 and chain d6 are within 4%.
  - c=4 and c=16: chain d6.
  - c=64: chain d3.
- **The roofline model from section 3 picks the right family at c ≥ 16.**
- **At c=1 the model's predicted tree win does not appear.** A tree step costs 28 to 32 ms vs 24 ms for chain d6. The
  target forward accounts for only about 3 ms of that. The rest is our tree-mode overhead, mainly the draft
  without CUDA graphs. With it removed, k4 d6 would be about 4.66 / 27 ms ≈ 170 tok/s, about +17% over the best chain.

### 5. Same sweep on stock SGLang EAGLE-3 (`OUT=runs/opt_stock run_opt.sh stock...`, `runs/opt_stock/`)

Stock flags only, no HSD/RS flag. Stock trees are pruned to the N best nodes by draft score,
verified by SGLang's stock rule, with draft CUDA graphs. Per-request / total tok/s:

| config | N | accept | c=1 | c=4 | c=16 | c=64 |
|---|---|---|---|---|---|---|
| chain d3 | 4 | 2.76 | 131.5 | 122.2 / 480 | 98.3 / 1354 | **41.9 / 2173** |
| chain d6 | 7 | 3.40 | 142.1 | 127.3 / 504 | **99.5 / 1378** | 36.3 / 2075 |
| tree k4 d6 N16 | 16 | 4.39 | 176.0 | **148.5 / 570** | 67.0 / 961 | 20.4 / 1180 |
| tree k8 d5 N32 | 32 | 4.45 | 176.9 | 137.8 / 546 | 44.8 / 667 | 12.6 / 742 |
| tree k10 d5 N60 | 60 | 4.67 | **177.1** | 80.5 / 314 | 24.4 / 365 | 7.0 / 416 |
| tree k4 d6 N85 | 85 | 4.81 | 166.5 | 66.2 / 262 | 19.9 / 302 | 5.7 / 341 |

- **EAGLE-3 trees have the same problem.** They beat the chain only up to c≈4. From c=16 the chain wins, and its 85-node tree
  collapses like ours (5.7 vs 5.2 tok/s per request at c=64). Above c=1, the roofline model picked the
  measured winner every time.
- **EAGLE's edge is pruning.** It gets 4.39 tok/step from 16 nodes, while our unpruned sampled trees need 23 nodes
  for 4.23 and 85 for 4.66. So its small tree still wins at c=4.
- **Its tree step is cheap.** N16 costs about 1 ms more per step than its chain (24.9 vs 23.9 ms). So its +25% acceptance
  turns into +24% speed at c=1. Ours turns into +4%.
- **HSD helps chains slightly:** chain d6 is 3.51 with HSD vs 3.40 stock.
- **Caveat: these "stock" runs used our branch's build with the HSD/RS flags off.** All our changed paths sit
  behind those flags (checked in `eagle_worker_v2.py` / `eagle_utils.py`). For a PR-grade A/B,
  re-run them on `/data/twang/sglang_pristine` (about 25 min).

### 6. Same N: tree vs chain at equal verification cost (`runs/opt_sameN/`)

Stock EAGLE-3 trees pruned to the chain's own N. Total tok/s at c=4 / 16 / 64:

| | accept | c=1 | c=4 | c=16 | c=64 |
|---|---|---|---|---|---|
| chain d6 (N=7) | 3.52 | 146.7 | 503 | 1266 | 1809 |
| tree k4 d6, N=7 | 3.75 | 153.1 | 525 | 1348 | 1970 |
| chain d3 (N=4) | 2.72 | 130.0 | 486 | 1428 | 2461 |
| tree k4 d3, N=4 | 2.86 | 134.4 | 433 | 1327 | 2325 |

- **At equal N a tree accepts 3–7% more and is about as fast.** Trees are not inherently worse;
  only *oversized* trees are.
- **Noise:** run-to-run, c=1 varies about ±3% and c ≥ 16 up to ±13%. This was measured by repeating the
  chains from section 5.

### 7. Oracle study: best stock EAGLE-3 shape per load and context (`hsd_eval/run_oracle.sh`)

Setup:
- `oracle_client.py` drives SGLang's native `/generate` at a fixed concurrency.
- GSM8K prompts, optionally with a shared K-shot prefix to set the context length.
- `max(24, 3c)` requests per point.
- Records `spec_verify_ct`, so acceptance is measured per request.
- Server: `--mem-fraction-static 0.7 --max-running-requests 64`.

`analyze_oracle.py` summarizes `runs/oracle/`. Best config per point (total tok/s):

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| context ~115 tokens | tree N32: 178 | tree N32: 327 | tree N16: 542 | tree N7: 835 | chain d3: 1306 | chain d3: 1614 | chain d3: 1844 |
| context ~1.85k tokens | tree N60: 177 | tree N16: 325 | tree N16: 597 | tree N16: 804 | chain d3: 1376 | chain d3: 1696 | chain d3: 1833 |

How much each fixed config loses against the per-point best (short context):

| config | c=1 | c=4 | c=16 | c=64 | mean |
|---|---|---|---|---|---|
| chain d3 | −29% | −25% | 0% | 0% | −16% |
| tree N16 | −4% | 0% | −23% | −36% | −17% |
| tree N32 | 0% | −18% | −49% | −62% | −32% |
| tree N60 | −4% | −41% | −72% | −78% | −50% |

- **No fixed shape is close to best everywhere.** Adapting the shape to load would gain:
  - +25 to 49% at c ≤ 8 over the best fixed config (chain d3)
  - 2.6 to 4.5× at c ≥ 16 over tree-first configs (N32/N60)
- **The best shape shrinks as concurrency grows,** from big tree to small tree to short chain. That is the budget rule of section 3.
- **Longer context keeps trees optimal to higher concurrency.** N16 is still best at c=8 with 1.85k context. Attention
  memory traffic enlarges the free budget.
- **Drafter context limit.** The lmsys EAGLE-3 head was trained for 2048 positions (`max_position_embeddings`
  2048 in its config). Past that, acceptance collapses. For tree N16 at c=1:

  | prompt tokens | 115 | 1380 | 1846 | 3349 | 5016 |
  |---|---|---|---|---|---|
  | accept | 4.45 | 4.65 | 4.48 | 1.90 | 1.64 |

  - Speed falls to about plain decoding, and SGLang gives no warning.
  - The study is therefore capped at 1.85k context.
  - A controller should watch acceptance per request and fall back.
- **Caveats:**
  - One run per point with 24 to 192 requests, so cells are noisy (about ±15%).
  - The oracle is an upper bound; a real controller must switch shapes without stalling.

**Idea derived from this: load-aware speculation in SGLang.** Choose the draft shape at each step from:
- the batch size and context length
- a cost model of the free verification budget
- live acceptance

Then capture a CUDA graph per shape. Prior work picks shapes offline (Sequoia) or adapts chain
length only (SmartSpec). Status: prototype in progress (see section 8).

### 8. Adaptive speculation prototype: tree-aware `--speculative-adaptive` (branch `SGLang-adaptive-tree`)

SGLang already has adaptive speculative decoding (`--speculative-adaptive`, `speculative/adaptive_*.py`):
- It pre-builds one runtime state (CUDA graphs and attention backends) per draft depth.
- It routes each batch to a per-batch-size slot, with an EMA of acceptance choosing the depth.
- Its limits: chains only (`topk == 1` was rejected, N hard-wired to `steps + 1`). Its default config turns speculation
  off at batch 64.

The change is uncommitted, on branch `SGLang-adaptive-tree`, cut from base `31e55a5120`; 6 files, +39/−18:
- **`draft_tokens` map:** the adaptive config may carry a map, e.g. `{"6": 32, "5": 16, "3": 4}`. Each depth's state is then a
  score-pruned tree of N tokens. Missing entries default to `steps + 1`, so old configs are unchanged.
- **`topk > 1` allowed.**
- **Buffer sizing** uses the largest configured N.
- **Tests:** all adaptive/spec unit tests pass (436).

Policy tested: `hsd_eval/configs/adaptive_tree_k4.json`, topk=4:

| running requests | state |
|---|---|
| 1–3 | steps 6, N=32 |
| 4–15 | steps 5, N=16 |
| 16+ | steps 3, N=4 |

Results on the oracle grid: total tok/s in `runs/oracle/adaptive_*`, `analyze_oracle.py`.

| context, mode | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 | mean loss vs oracle |
|---|---|---|---|---|---|---|---|---|
| ~115 tokens, SGLang adaptive (chains, default) | 138 | 255 | 477 | 583 | 1101 | 1469 | 1881 | −18% |
| ~115 tokens, **tree-aware adaptive** | **176** | **339** | **610** | **829** | 1171 | 1567 | 1699 | **−4%** |
| ~1.85k tokens, SGLang adaptive | 140 | 254 | 506 | 567 | 1406 | 1917 | 1832 | −13% |
| ~1.85k tokens, **tree-aware adaptive** | **175** | **330** | **549** | 617 | 1014 | 1398 | 1795 | −13% |

- **Low concurrency (c ≤ 4–8): within 0 to 2% of the oracle,** and +28 to 42% over SGLang's adaptive mode.
- **High concurrency (c ≥ 16): up to 28% behind SGLang's chain adaptive,** worst at long context. The probable cause is that
  the "chain" state is a topk=4 tree pruned to N=4. That skips SGLang's topk=1 chain fast path (`build_chain_tree`,
  `_rebuild_topk1_chain_buffers`), and it matches the −6 to −11% of tree k4 d3 N4 vs chain d3 in section 6.
- **Update 2026-11-18: per-state topk and budget-derived slots, with 3 repeats.**
  Run with `run_oracle_reps.sh` → `runs/oracle_r{1,2,3}`; analyze with `analyze_oracle_reps.py`.
  - **New config keys:** `draft_topk` (steps → k) sits alongside `draft_tokens`. The draft extend always carries the startup top-k,
    and each state slices it.
  - **Policy:** `configs/adaptive_tree_budget.json`. Slots are chosen so that bs × N ≤ ~110:

    | running requests | state |
    |---|---|
    | 1–3 | k4, steps 6, N32 |
    | 4–7 | k4, steps 5, N16 |
    | 8–15 | k4, steps 4, N7 |
    | ≥16 | k1 chain, steps 3 |

  - **Results:** mean total tok/s over 3 repeats; oracle = best fixed config by mean.

    | context | mode | c=1 | c=4 | c=16 | c=64 | mean vs oracle |
    |---|---|---|---|---|---|---|
    | ~115 tokens | SGLang adaptive | 143 | 423 | 998 | 1807 | −16% |
    | ~115 tokens | **budget tree policy** | **183** | **525** | **1128** | 1790 | **−3%** |
    | ~1.85k tokens | SGLang adaptive | 138 | 469 | 880 | 1795 | −16% |
    | ~1.85k tokens | **budget tree policy** | **186** | **601** | **1098** | **1972** | **+1%** |

  - **Against SGLang's adaptive mode:** +24 to 35% at c ≤ 4, +7 to 25% at c = 8 to 32, −1 to +10% at c=64.
  - **Verified tokens per pass:** 32 to 67 up to c=16 for ours, vs 4 to 35 for SGLang's. SGLang leaves the free budget unused.
  - **Weak spot:** c=4 to 8, at −10 to −17%. States are keyed by depth, so the N16 and N7 states use d=5 and d=4,
    while the best fixed shapes use d=6.
  - **Noise:** spread is about ±10 to 20% at c ≥ 8.
- **Next:**
  - key states by (steps, N, topk), so the shapes at c=4 to 8 can be the depth-6 trees
  - hysteresis at slot boundaries (the log shows the state flipping every step when the batch hovers at 3 or 4)
  - repeated runs (cells are about ±15%)
  - a GSM8K accuracy check with state switching mid-request

### 9. Roofline check and profiling of stock EAGLE-3 trees (`roofline_check.py`, `runs/profile/`)

**Setup.** Measured ceilings on this GPU (`hw_peaks.py`):
- **memory read:** 936 GB/s
- **bf16 GEMM:** about 120 TFLOPS sustained (115 to 165 depending on M)

The ideal step is draft + verify, each pass max(memory, compute). Measured step = Σ latency / Σ verify steps.

**Efficiency (ideal / measured), short context:**

| case | c=1 | c=4 | c=16 | c=64 |
|---|---|---|---|---|
| target-only | 98 to 101% | | | 76% |
| chains | 93 to 94% | 86 to 89% | 70 to 76% | 57 to 68% |
| trees | 81 to 91% | 61 to 83% | 63 to 75% | 61 to 86% |

**Profiles** (`profile_points.sh`, `/start_profile` under load, 30 steps; `trace_breakdown.py`):

| case | step | ideal | GEMM | small kernels | GPU idle |
|---|---|---|---|---|---|
| tree N32, c=1 | 25.8 ms | 21.7 ms | 21.3 ms (at roofline) | 2.3 ms (~900 launches/step) | 2.2 ms |
| tree N32, c=4 | 31.4 ms | | 24.4 ms | 4.2 ms | 2.8 ms |
| chain d3, c=64 | 39.8 ms (89% of ideal) | | | | |

- **Chain d3, c=64:** its larger measured gap (62.5 ms per step) comes from prefill interference, which is not tree-specific.
- **Where the idle time comes from:** CPU-side attention planning with blocking D2H reads.
  - Verify: FlashInfer `plan()` costs about 0.75 ms/step.
  - Draft: `common_template` costs about 0.45 ms/step, including `kv_indptr.cpu()`.

**Improvement made** (branch `SGLang-adaptive-tree`, `flashinfer_backend.py`): the sync-free verify plan
(`SGLANG_OPT_FLASHINFER_VERIFY_HOST_PLAN`, opt-in) was gated to `topk == 1`. The gate is relaxed to any topk,
because a tree's mask is packed per replay with the same per-request layout. `test_flashinfer_host_plan.py`
already covers arbitrary masks. The existing opt-in `SGLANG_OPT_FLASHINFER_DRAFT_HOST_INDPTR` already
supports trees.

Results with both flags on (`ab_test.sh`, `runs/ab/`):

| | flags off | both flags on |
|---|---|---|
| greedy outputs (16 prompts, each flag separately) | | identical, 16/16 |
| GPU idle, c=1 | 2.18 ms/step | 0.79 ms/step |
| step time, c=1 | 25.84 ms | 24.43 ms (−5.5%) |
| GPU idle, c=4 | 2.80 ms/step | 1.19 ms/step |
| step time, c=4 | 31.41 ms | 30.71 ms (−2.2%) |
| end-to-end tok/s (2 runs each) | | +0 to 3%, within run noise |

**Follow-up (2026-11-18): three system steps.**

1. **Kept.** Both host-plan flags default **on for EAGLE trees** (topk > 1) when unset; an explicit
   setting still wins. Implemented via `_host_plan_enabled` in `flashinfer_backend.py`, using
   `EnvBool.is_set()`. Chains are unchanged.
2. **Rejected and reverted.** EAGLE verify attention without split-KV removed the merge kernel
   (−0.4 ms/step at c=4), but:
   - long context (~1.85k tokens) at c=1 lost 19% (183 → 149 tok/s): one request's attention ran on too few CTAs
   - greedy matched on only 7/16 prompts: bf16 reduction order changes, flipping near-ties
3. **Rejected and reverted.** Reusing a zeroed `draft_probs` scratch in target-only tree sampling,
   instead of a per-step `zeros_like` of [bs, N, vocab]:
   - exact: 300 random kernel trials and greedy 16/16
   - no measurable gain on tree N60 at c=16/64: the zero-fill is about 0.3% of a step there,
     and about 0.2% at c=4
   - fusing softmax and top-p for verify would save at most about 1.5% at c=4; not pursued

**Conclusion.** With the flags on, the GPU is busy for 96 to 97% of a tree step. The remaining gap to the
ideal is GPU work:
- per-layer small kernels, which the ideal omits
- tree-masked verify attention with split-KV merge (about 1.6 ms at c=4)
- verify sampling over all nodes, i.e. softmax plus top-p over the 128k vocab (about 0.5 ms at c=4)
- GEMMs near the ridge at c=4

Large further gains need less work per step (adaptive shape, pruning), not less overhead.

### Recommended draft shape on this GPU (HSD), for now

| running requests | shape | flags |
|---|---|---|
| 1–2 | tree k2–3, d6 (clearly best once the draft CUDA-graph fix lands) | `--speculative-eagle-topk 2 --speculative-num-steps 6 --speculative-num-draft-tokens 23 --speculative-use-hsd` |
| 4–16 | chain d6 | `--speculative-eagle-topk 1 --speculative-num-steps 6 --speculative-num-draft-tokens 7 --speculative-use-hsd` |
| ~64 | chain d3 | `--speculative-eagle-topk 1 --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-use-hsd` |

No single fixed config wins at all concurrencies, for stock EAGLE-3 or for us.

### Reference code without SGLang (paper's EAGLE-3H + tree-hsd)

Two runs, both batch 1:
- `hsd_eval/ref_bench.py`: the paper's own benchmark, 60 questions over 6 tasks, T=1, `runs/ref/`.
- `hsd_eval/ref_gsm8k.py` + `score_ref_gsm8k.py`: GSM8K with the same prompt and grader as sgl-eval. The scorer reproduces sgl-eval's
  82.4867% exactly on SGLang outputs.

`ref_bench.py` results (tok/step / tok/s):

| config | eagle (top-k) | chain RRSw | chain HSD | sampled RRSw | sampled HSD | hybrid1 RRSw | hybrid1 HSD (fused) |
|---|---|---|---|---|---|---|---|
| tok/step | 4.07 | 3.19 | 3.40 | 4.16 | 4.24 | 4.18 | 4.37 |
| tok/s | 117.1 | 102.2 | 108.7 | 108.7 | 110.2 | 109.2 | 114.3 |

The full 1319-question reference GSM8K run (`run_ref_gsm8k.sh`) was stopped at about 211/660 questions
of shard 0. `runs/ref_gsm8k/shard0.jsonl` resumes if restarted.

## How the experiments are run

There is no custom benchmark harness. The stack has three layers:

1. **Server: SGLang from this branch, configured only by CLI flags.** `hsd_eval/launch_server.sh` runs
   `python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
   --speculative-algorithm EAGLE3 --speculative-draft-model-path
   lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B --dtype bfloat16 --context-length 8192`, plus the
   config's flags.
   - Stock configs use SGLang's documented `--speculative-*` options.
   - HSD configs add our `--speculative-use-hsd`.
   - Each run writes the exact flags to `flags.txt`.
2. **Client: sgl-eval 0.1.3** (`/data/twang/sgl_eval_venv`). It sends OpenAI-compatible
   `/v1/chat/completions` requests:
   `sgl-eval run gsm8k --base-url http://127.0.0.1:PORT/v1 --num-examples N --num-threads C
   --temperature 0.6 --top-p 0.9 --max-tokens 1024 --out-dir ...`.
   - Concurrency = `--num-threads`.
   - The first N questions are taken in order.
   - sgl-eval grades the answers itself.
   - It writes `output-rs0.jsonl` (per-answer text, tokens and timestamps) and `metrics.json`.
3. **Scripts:** `run_opt.sh` loops over configs. For each one it starts a fresh server, runs sgl-eval at c=1/4/16/64,
   saves `GET /server_info` after c=1 (for the acceptance counter), then kills the server.
   `analyze_opt.py` builds the tables and the roofline pick.
   Other runners:
   - `run_configs.sh`: full set, c=64, plus distribution samples
   - `run_gsm8k_bs1.sh`: full set, c=1
   - `run_hsd100.sh`: matched 100-question comparison
   - `run_gsm8k_c64.sh`: stock-tree c=64 control, unused

Evaluation pitfalls we hit:
- **Read the depth from both flags.** SGLang counts the root in `--speculative-num-draft-tokens`, so a chain with 6 draft tokens drafts
  only 5. Match depth when comparing chain vs tree.
- **The accept counter is cumulative.** `avg_spec_accept_length` counts from server start, so use a fresh server per measurement
  (or read it right after the run).
- **Per-request speed ≠ total throughput.** At c=64 the old tables showed total throughput (about 2000 tok/s). Per request it is about 31 to 36 tok/s.
- **`max-running-requests` puts a queue in front of c=64 latency.** Keep it equal across configs being compared.
- **`pkill -f <pattern>` can match the shell running it.** Use the `[-]-port` bracket trick. This killed runs twice.
- **Make sure the port is free before starting a server.** A leftover server on the port silently answers for the next config, so the runners
  check `/health` first.

## Method directions (not started)

EAGLE-3's own change is the drafter (multi-layer feature fusion, "training-time test"); we already
use an EAGLE-3 head. The tree builder comes from EAGLE-2:
- EAGLE-2 picks the top-k children, expands by path score, and then **reranks and prunes to N**.
- We cannot prune freely: dropping drawn siblings breaks the without-replacement property that HSD/RRSw rely on.

Ideas, by cost and benefit:

1. **Pruned sampled trees.** Keep, per node, only a *prefix in draw order* of its sampled children.
   That is equivalent to drawing fewer children.
   - Choose the prefix length from draft-only information, e.g. EAGLE-2-style scores, so that about 16 to 30 nodes carry most of the
     acceptance.
   - This should be lossless, since the walk tries siblings in order and falls back to the residual. **Prove it first with
     `exact_real.py` / `verify_sampled_tree_wor.py`.**
   - Goal: stock-N16-like budgets (4.4 tok/step at 16 nodes) plus HSD's gain.
2. **Draft temperature as a free knob.** Any q is lossless if verification uses the q actually sampled from. A lower
   draft temperature makes sampled children closer to top-k.
3. **Hybrid/spine trees** (open item 2 below): this is where HSD beats RRSw.
4. **Batch-aware shape.** Choose k, d and N per batch size from the budget above.
5. **Train the head for HSD acceptance.** Use a TV/overlap loss instead of cross-entropy, as DistillSpec does, with EAGLE-3-style
   training-time test on the sampled-tree walk. This is the largest effort.

Related work to compare against: "Traversal Verification" (2025), a leaf-to-root lossless tree
verification. We have not checked its details.

## Environment (the non-obvious part)

SGLang main is CUDA-13-only (torch 2.14), but this box has driver 570 (CUDA 12.8) and no
sudo. **Always `source hsd_eval/env.sh` first**; it sets up:
- conda env `sglang_hsd` (py3.12, `pip install -e sglang/python` built with
  `SGLANG_BUILD_RUST_EXTS=none`, since there is no cargo);
- the CUDA 13.0 forward-compat `libcuda` in `/data/twang/cuda13_compat` (without it,
  `torch.cuda.is_available()` is False);
- nvcc 13.0 from the env's pip wheels (`nvidia-nvvm` / `nvidia-cuda-crt` must stay pinned
  to 13.0.*, because 13.5 emits PTX that ptxas 13.0 rejects);
- gcc-13 via the shim `/data/twang/gcc13_bin` (JIT kernels need C++20; the default is gcc-9).

Model gotchas, all handled in `hsd_eval/launch_server.sh`:
- `HF_HUB_OFFLINE=1`: the Llama repo is gated but cached.
- `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`: the head's config says a 2048 context.
- Explicit `--dtype bfloat16`: the head's config says fp16, and `auto` crashes in RMSNorm.
- Prompts must go through the tokenizer chat template. Without the default system header,
  acceptance drops by about 10%.
- `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` does not load on SGLang main.

A pristine `HEAD` worktree for A/B test runs is at `/data/twang/sglang_pristine`; remove it
with `git -C sglang worktree remove /data/twang/sglang_pristine` when no longer needed.

## Open items before the PR

1. **Broaden accuracy coverage.** The checks above all pass; next, more prompts and tasks
   (the paper uses MT-bench, HumanEval, GSM8K, Alpaca, CNN/DM, NQ) and batch sizes above 1
   for the distribution test.
2. **Hybrid/spine trees.** Reserve one expansion per level for the first-child path (the
   reference's `topK_genrate(spines=1, extra=e)`); this is where HSD beats RRSw. It needs
   a change in `select_top_k_tokens` and must keep the q-row layout contract above.
3. **Memory.** The per-node fp32 `target_probs` [bs, 85, vocab] and q rows are not in
   SGLang's static memory budget, so a tree at bs 48 OOMs. The eval caps
   `--max-running-requests 32 --mem-fraction-static 0.6`. This needs either reserving the
   memory or a design that does not materialize full rows.
4. **Speed.** It is measured now (see Performance results). In priority order:
   - **Draft CUDA graphs are disabled in tree mode,** because their q buffers are chain-shaped
     (see the `TODO(perf)` in `eagle_worker_v2._capture_cuda_graphs`). This is the main cost at c=1. A tree step is
     about 4 to 8 ms slower than a chain step, while stock EAGLE's pruned tree costs about 1 ms more.
   - **Pruned sampled trees** (Method directions, item 1). Our unpruned trees waste about 94% of
     their verified nodes, which caps tree HSD at c ≤ 2 on this GPU.
   - **Batch-size-dependent draft shape** (tree at low concurrency, chain at high).
   - **The verifier is minor:** about 0.17 ms per step at bs=1. It runs one program per request with several
     full-vocab passes per path. The reference `hsd_fused` parallelises each path over (position × vocab block).
5. **Determinism.** Tree mode rejects `--enable-deterministic-inference` and ignores
   per-request seeds for node expansions, because seeded draws hash (request, position)
   and siblings share both. A seed salted by node index would fix this.
6. **Triton race pitfall.** A scalar `tl.store` is issued by one thread, so other threads
   can read stale values. Two such races (path scratch read-back; load-then-zero of
   `prop[x]`) caused intermittent illegal accesses and an infinite loop. Any new scratch
   traffic in the kernel needs `tl.debug_barrier()` between a scalar store and later
   loads. The helper `_remove_token` shows the pattern.
7. **Upstream side findings**, maybe worth separate issues: the fp16 EAGLE-3 head crashes
   with `--dtype auto` on a bf16 target; and the failing dflash unit test on HEAD.
8. **Clean stock baseline.** Re-run the stock EAGLE-3 sweep on the pristine worktree for the PR,
   and run the paper's full GSM8K reference run if a no-SGLang comparison is needed.
9. **PR hygiene.** Follow `sglang/.claude/rules` (comment style, naming skill
   `speculative-naming`, test admission), add docs for the flag, and push `SGLang-tree-hsd`.
   Maybe split into two PRs: (a) the sampled tree + RRSw, (b) tree HSD.
10. **Adaptive-tree accuracy (branch `SGLang-adaptive-tree`, 2026-11-18). Open: a first-token shift
    in adaptive mode.** Details: `adaptive_tree_eagle3.md`, Accuracy section. Data: `hsd_eval/runs/accuracy*`.
    Scripts: `dist_followup.sh`, `dist_control.sh`.
    - **Task accuracy is unchanged:** GSM8K at T = 0, 0.3, 0.6 and 1.0 is within noise of target-only.
    - **The test is calibrated:** two independent 20k-sample target-only runs agree (p = 0.55).
    - **Adaptive-mode servers sometimes sample the first output token from a sharper distribution**
      (top token 53.4% vs 50.2%; at 20k samples, Bonferroni p = 4e-5). The first token comes from the
      target's prefill, before any drafting.
      - Seen in: our tree config, the chain-only adaptive config, and the T=0.5 run.
      - Not seen in: one 3k-sample adaptive-tree run, target-only, or the stock fixed tree.
      - So it is in the adaptive path and not tree-specific. Stock SGLang adaptive vs our branch is unresolved.
    - **Next:** run 20k samples of SGLang adaptive mode on base `31e55a5120`; vary the prefill CUDA
      graph and the radix cache. Separately, the stock fixed tree shows a weak signal at positions 3–4
      (raw p ≈ 1e-3); rerun it at 40k samples.
