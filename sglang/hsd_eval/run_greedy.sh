#!/usr/bin/env bash
# Check (1) on GPU 1: target_only (with logprobs), chain_hsd, tree_hsd, tree_rrsw.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
out=runs/greedy; mkdir -p $out
TREE="--speculative-num-steps 6 --speculative-eagle-topk 4 --speculative-num-draft-tokens 85 --mem-fraction-static 0.6 --max-running-requests 32"
CHAIN="--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
wait_up() { for _ in $(seq 240); do curl -sf http://127.0.0.1:30001/health > /dev/null && return 0; sleep 5; done; return 1; }
run() {  # name, extra args for check_greedy, server launch...
  local name=$1 extra=$2; shift 2
  "$@" > $out/server_$name.log 2>&1 &
  wait_up && python check_greedy.py --port 30001 $extra --out $out/$name.json > $out/$name.log 2>&1
  pkill -f -- "[-]-port 30001 "; sleep 15; echo "$name done"
}
run target_only --logprobs env CUDA_VISIBLE_DEVICES=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 --mem-fraction-static 0.75 --port 30001 --return-output-ids
run chain_hsd "" env GPU=1 PORT=30001 bash launch_server.sh $CHAIN --speculative-use-hsd --return-output-ids
run tree_hsd  "" env GPU=1 PORT=30001 bash launch_server.sh $TREE --speculative-use-hsd --return-output-ids
run tree_rrsw "" env GPU=1 PORT=30001 bash launch_server.sh $TREE --speculative-use-rejection-sampling --return-output-ids
echo "greedy all done"
