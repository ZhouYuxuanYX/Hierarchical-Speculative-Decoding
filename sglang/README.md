# HSD in SGLang

Tree HSD and RRSw verification for EAGLE-3 sampled draft trees, implemented in SGLang, plus the
evaluation that checks them.

| file | what it is |
|---|---|
| [`SGLang-tree-hsd.diff`](SGLang-tree-hsd.diff) | the SGLang change (12 files, +865 / −28) |
| [`handover.md`](handover.md) | full write-up: design, accuracy (lossless), performance, open items |
| [`hsd_eval/`](hsd_eval/) | scripts that run every experiment, and their summary results (`hsd_eval/runs/`) |

## Applying the patch

The patch is against SGLang commit `31e55a5120`:

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 31e55a5120 && git apply ../SGLang-tree-hsd.diff
```

Flags: `--speculative-use-hsd` (HSD) or `--speculative-use-rejection-sampling` (RRSw). With
`--speculative-eagle-topk > 1`, both run on a sampled draft tree. The commands for each config are in
`hsd_eval/run_configs.sh`, and the environment notes are in `hsd_eval/README.md`.

## What `hsd_eval/runs/` holds

Summary results only: per-run JSON results, sgl-eval `metrics.json`, and distribution-test samples.
Server logs, per-question generations and profiler traces are not included.
