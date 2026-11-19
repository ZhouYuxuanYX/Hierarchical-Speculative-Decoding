# HSD in SGLang: accuracy evaluation

Code under test: `../sglang`, branch `tree-hsd` (SGLang main + sampled-tree HSD/RRSw).
Model: Llama-3.1-8B-Instruct with `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`, TP=1, bf16.

## Environment

`source env.sh` (conda env `sglang_hsd`). This box has driver 570 (CUDA 12.8) and SGLang
main needs CUDA 13, so env.sh adds the CUDA 13.0 forward-compat libcuda, nvcc 13.0 from
pip wheels and gcc-13. `launch_server.sh` wraps `sglang.launch_server` with the same
environment and the model flags.

## Configs (`run_configs.sh`)

| name | flags |
|---|---|
| target_only | no speculative decoding (reference) |
| chain_token | EAGLE-3 chain, 5 steps, `--speculative-use-rejection-sampling` (token-wise) |
| chain_hsd | same chain, `--speculative-use-hsd` (= block verification) |
| tree_rrsw | sampled tree, topk 4, 6 steps, 85 nodes, `--speculative-use-rejection-sampling` |
| tree_hsd | same tree, `--speculative-use-hsd` |
| lossy_ctrl | positive control: target-only tree verify with accept thresholds 0.3 (not lossless) |

Tree configs add `--mem-fraction-static 0.6 --max-running-requests 32`: the per-node fp32
probability rows are not part of SGLang's static memory budget yet.

```bash
GPU=0 PORT=30000 bash run_configs.sh target_only chain_token tree_hsd
GPU=1 PORT=30001 bash run_configs.sh chain_hsd tree_rrsw lossy_ctrl
python analyze.py runs
```

Per config: GSM8K with sgl-eval 0.1.3 (all 1319, T=0.6, top-p 0.9, max 1024 tokens; the
model's generation_config settings), the server's average accept length over that run,
then 3000 samples of the first 8 tokens at T=1 for two high-entropy prompts.

## Losslessness test (`analyze.py`)

For each prompt, the token distribution at each of the 8 positions and the joint of the
first two tokens are compared with `target_only` by a two-sample chi-square test; the
reported p is the minimum over the 18 tests, Bonferroni-corrected. `lossy_ctrl` checks
that the test can see a deviation end to end.

Kernel-level exact checks live in
`../sglang/test/registered/kernels/ops/speculative/test_sampled_tree_verify.py`.
