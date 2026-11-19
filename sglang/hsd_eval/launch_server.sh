#!/usr/bin/env bash
# Usage: GPU=0 PORT=30000 bash launch_server.sh <extra sglang flags...>
# Llama-3.1-8B-Instruct + lmsys EAGLE-3 head (the HSD paper's setup), TP=1.
# --dtype bfloat16: the head config says float16, the target bfloat16; "auto" mixes them.
set -euo pipefail
export LD_LIBRARY_PATH=/data/twang/cuda13_compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}  # CUDA 13 on the 570 driver
export HF_HUB_OFFLINE=1
export CUDA_HOME=/home/donglab-gpu-local2/anaconda3/envs/sglang_hsd/lib/python3.12/site-packages/nvidia/cu13  # nvcc 13.0 for FlashInfer JIT
export PATH=$CUDA_HOME/bin:/data/twang/gcc13_bin:$PATH  # gcc-13: JIT kernels need C++20
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1  # the EAGLE-3 head's config says 2048
export CUDA_VISIBLE_DEVICES=${GPU:-0}
exec /home/donglab-gpu-local2/anaconda3/envs/sglang_hsd/bin/python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --dtype bfloat16 --context-length 8192 --mem-fraction-static 0.75 \
  --port ${PORT:-30000} "$@"
