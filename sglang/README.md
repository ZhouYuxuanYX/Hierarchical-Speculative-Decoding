# HSD and adaptive tree drafting in SGLang

Two independent SGLang patches, both against SGLang commit `31e55a5120`, plus the evaluation:

1. **Tree HSD / RRSw:** lossless verification of EAGLE-3 sampled draft trees.
2. **Adaptive tree drafting:** `--speculative-adaptive` states can be pruned EAGLE trees, so the
   draft shape follows the load (wide tree at low concurrency, chain at high). Up to +35% throughput
   over SGLang's chain-only adaptive mode on an RTX 6000 Ada.

| file | what it is |
|---|---|
| [`SGLang-tree-hsd.diff`](SGLang-tree-hsd.diff) | the SGLang change (12 files, +865 / −28) |
| [`SGLang-adaptive-tree.diff`](SGLang-adaptive-tree.diff) | the adaptive-tree change (11 files, +170 / −29) |
| [`adaptive_tree_eagle3.md`](adaptive_tree_eagle3.md) | adaptive tree write-up: what changed, results, accuracy, diff walkthrough |
| [`handover.md`](handover.md) | full write-up: design, accuracy (lossless), performance, open items |
| [`hsd_eval/`](hsd_eval/) | scripts that run every experiment, and their summary results (`hsd_eval/runs/`) |

## Applying the patch

The patch is against SGLang commit `31e55a5120`:

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 31e55a5120 && git apply ../SGLang-tree-hsd.diff   # or ../SGLang-adaptive-tree.diff
```

The two patches are independent. Apply one per checkout.

Flags: `--speculative-use-hsd` (HSD) or `--speculative-use-rejection-sampling` (RRSw). With
`--speculative-eagle-topk > 1`, both run on a sampled draft tree. The commands for each config are in
`hsd_eval/run_configs.sh`, and the environment notes are in `hsd_eval/README.md`.

Adaptive tree: `--speculative-adaptive --speculative-adaptive-config hsd_eval/configs/adaptive_tree_budget.json
--speculative-eagle-topk 4 --speculative-num-steps 6`. See `adaptive_tree_eagle3.md`, "Running it".

**Accuracy caveat (adaptive tree):** GSM8K matches target-only at every temperature tested, but a
20k-sample distribution test shows an open first-token shift in adaptive mode. See the Accuracy
section of `adaptive_tree_eagle3.md`.

## What `hsd_eval/runs/` holds

Summary results only: per-run JSON results, sgl-eval `metrics.json`, and distribution-test samples.
Server logs, per-question generations and profiler traces are not included.
