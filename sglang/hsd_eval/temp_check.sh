#!/usr/bin/env bash
# Temperature sweep: GSM8K (c=8, T in 0/0.3/1.0, top-p 1.0) + switching distribution test at T=0.5.
# Usage: GPU=0 PORT=30000 bash temp_check.sh <name> <extra server flags...>
set -uo pipefail
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1; export HF_HUB_OFFLINE=1
GPU=${GPU:-0}; PORT=${PORT:-30000}; name=$1; shift
COMMON="--mem-fraction-static 0.7 --max-running-requests 64 --return-output-ids"
out=runs/accuracy_temp/$name; mkdir -p $out
if [ "$name" = target ]; then
  CUDA_VISIBLE_DEVICES=$GPU SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 $COMMON --port $PORT > $out/server.log 2>&1 &
else
  GPU=$GPU PORT=$PORT bash launch_server.sh "$@" $COMMON > $out/server.log 2>&1 &
fi
server=$!
for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $server 2>/dev/null || break; sleep 5; done
curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo "$name: server failed"; exit 1; }
for T in 0 0.3 1.0; do
  /data/twang/sgl_eval_venv/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:$PORT/v1 --num-examples 1319 \
    --num-threads 8 --temperature $T --top-p 1.0 --max-tokens 1024 --out-dir $PWD/$out/gsm8k_T$T > $out/gsm8k_T$T.log 2>&1
  echo "[$(date +%T)] $name T=$T $(grep -m1 'score' $out/gsm8k_T$T.log)"
done
python dist_mixed.py --port $PORT --temperature 0.5 --out $out/dist_T0.5.json > $out/dist.log 2>&1
pkill -f -- "[-]-port $PORT"; sleep 15
echo "[$(date +%T)] $name done"
