#!/usr/bin/env bash
# Usage: GPU=0 PORT=30000 bash run_configs.sh <config name>...
# For each config: fresh server -> GSM8K (sgl-eval) -> accept length -> distribution samples.
set -uo pipefail
cd "$(dirname "$0")"
source env.sh
GPU=${GPU:-0}; PORT=${PORT:-30000}
NUM_EXAMPLES=${NUM_EXAMPLES:-1319}
DIST_SAMPLES=${DIST_SAMPLES:-3000}
RUNS=${RUNS:-runs}
CHAIN="--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
# Per-node fp32 probability rows [bs, 85, 128256] are not in the static memory budget.
TREE="--speculative-num-steps 6 --speculative-eagle-topk 4 --speculative-num-draft-tokens 85 --mem-fraction-static 0.6 --max-running-requests 32"

flags_of() {
  case $1 in
    target_only)  echo "NOSPEC" ;;
    chain_token)  echo "$CHAIN --speculative-use-rejection-sampling" ;;
    chain_hsd)    echo "$CHAIN --speculative-use-hsd" ;;
    tree_rrsw)    echo "$TREE --speculative-use-rejection-sampling" ;;
    tree_hsd)     echo "$TREE --speculative-use-hsd" ;;
    # Positive control, lossy on purpose: relaxed acceptance of target-only tree sampling.
    lossy_ctrl)   echo "$TREE --speculative-accept-threshold-single 0.3 --speculative-accept-threshold-acc 0.3" ;;
    *) echo "unknown config $1" >&2; exit 1 ;;
  esac
}

for name in "$@"; do
  out=$RUNS/$name; mkdir -p $out
  flags=$(flags_of $name)
  echo "[$(date +%T)] $name: $flags"
  if [ "$flags" = "NOSPEC" ]; then
    CUDA_VISIBLE_DEVICES=$GPU SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 \
      --mem-fraction-static 0.75 --port $PORT --return-output-ids > $out/server.log 2>&1 &
  else
    GPU=$GPU PORT=$PORT bash launch_server.sh $flags --return-output-ids > $out/server.log 2>&1 &
  fi
  server=$!
  # Wait on /health only: any generate request would count in the accept length.
  for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
  curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo "$name: server did not start"; kill $server; continue; }
  /data/twang/sgl_eval_venv/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:$PORT/v1 \
    --num-examples $NUM_EXAMPLES --temperature 0.6 --top-p 0.9 --max-tokens 1024 \
    --out-dir $PWD/$out > $out/gsm8k.log 2>&1
  curl -s http://127.0.0.1:$PORT/server_info > $out/server_info_after_gsm8k.json
  python dist_sample.py --port $PORT --num-samples $DIST_SAMPLES --out $out/dist.json > $out/dist.log 2>&1
  kill $server; wait $server 2>/dev/null; pkill -f -- "[-]-port $PORT " ; sleep 10
  echo "[$(date +%T)] $name done: $(grep -m1 'score' $out/gsm8k.log)"
done
