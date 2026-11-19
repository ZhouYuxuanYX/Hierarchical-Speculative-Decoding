#!/usr/bin/env bash
# Higher-power losslessness check: 20k samples of prompt 1, tree HSD vs target-only, GPU 0.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
out=runs/big_prompt1; mkdir -p $out
wait_up() { for _ in $(seq 240); do curl -sf http://127.0.0.1:30000/health > /dev/null && return 0; sleep 5; done; return 1; }
GPU=0 PORT=30000 bash launch_server.sh --speculative-num-steps 6 --speculative-eagle-topk 4 --speculative-num-draft-tokens 85 \
  --mem-fraction-static 0.6 --max-running-requests 32 --speculative-use-hsd --return-output-ids > $out/server_tree_hsd.log 2>&1 &
wait_up && python dist_sample.py --port 30000 --prompts 1 --num-samples 20000 --out $out/tree_hsd.json > $out/tree_hsd.log 2>&1
pkill -f -- "[-]-port 30000 "; sleep 15
CUDA_VISIBLE_DEVICES=0 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 --context-length 8192 --mem-fraction-static 0.75 --port 30000 --return-output-ids > $out/server_target.log 2>&1 &
wait_up && python dist_sample.py --port 30000 --prompts 1 --num-samples 20000 --out $out/target_only.json > $out/target_only.log 2>&1
pkill -f -- "[-]-port 30000 "
echo "big_prompt1 done"
