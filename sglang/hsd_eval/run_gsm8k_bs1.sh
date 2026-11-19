#!/usr/bin/env bash
# GSM8K (all 1319, sgl-eval, T=0.6, top-p 0.9) at batch 1: one question in flight.
# Usage: GPU=0 PORT=30000 bash run_gsm8k_bs1.sh <config>...
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
GPU=${GPU:-0}; PORT=${PORT:-30000}
CHAIN="--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
TREE="--speculative-num-steps 6 --speculative-eagle-topk 4 --speculative-num-draft-tokens 85 --mem-fraction-static 0.6 --max-running-requests 32"
flags_of() {
  case $1 in
    target_only) echo "NOSPEC" ;;
    chain_token) echo "$CHAIN --speculative-use-rejection-sampling" ;;
    chain_hsd)   echo "$CHAIN --speculative-use-hsd" ;;
    tree_rrsw)   echo "$TREE --speculative-use-rejection-sampling" ;;
    tree_hsd)    echo "$TREE --speculative-use-hsd" ;;
    chain_hsd6)  echo "--speculative-num-steps 6 --speculative-eagle-topk 1 --speculative-num-draft-tokens 7 --speculative-use-hsd" ;;  # same draft depth as the tree (6)
    chain_token6) echo "--speculative-num-steps 6 --speculative-eagle-topk 1 --speculative-num-draft-tokens 7 --speculative-use-rejection-sampling" ;;
    eagle_tree)  echo "--speculative-num-steps 5 --speculative-eagle-topk 10 --speculative-num-draft-tokens 60" ;;
  esac
}
for name in "$@"; do
  out=runs/gsm8k_bs1/$name; mkdir -p $out
  # A leftover server on the port would silently answer for this config.
  if curl -sf http://127.0.0.1:$PORT/health > /dev/null; then echo "$name: port $PORT busy, abort"; exit 1; fi
  flags=$(flags_of $name)
  if [ "$flags" = "NOSPEC" ]; then
    CUDA_VISIBLE_DEVICES=$GPU SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 \
      --mem-fraction-static 0.75 --port $PORT > $out/server.log 2>&1 &
  else
    GPU=$GPU PORT=$PORT bash launch_server.sh $flags > $out/server.log 2>&1 &
  fi
  for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
  echo "[$(date +%T)] $name started"
  /data/twang/sgl_eval_venv/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:$PORT/v1 \
    --num-examples 1319 --num-threads 1 --temperature 0.6 --top-p 0.9 --max-tokens 1024 \
    --out-dir $PWD/$out > $out/gsm8k.log 2>&1
  curl -s http://127.0.0.1:$PORT/server_info > $out/server_info.json
  pkill -f -- "[-]-port $PORT"; sleep 15
  echo "[$(date +%T)] $name done: $(grep -m1 'examples (single' $out/gsm8k.log) $(grep -m1 'score' $out/gsm8k.log)"
done
