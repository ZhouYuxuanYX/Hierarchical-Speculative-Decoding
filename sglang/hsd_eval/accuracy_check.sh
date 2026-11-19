#!/usr/bin/env bash
# Losslessness checks for one config: sgl-eval GSM8K (c=8, c=64), greedy at c=8, switching distribution test.
# Usage: GPU=0 PORT=30000 bash accuracy_check.sh <name> <extra server flags...>
set -uo pipefail
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1; export HF_HUB_OFFLINE=1
GPU=${GPU:-0}; PORT=${PORT:-30000}; name=$1; shift
COMMON="--mem-fraction-static 0.7 --max-running-requests 64 --return-output-ids"
out=runs/accuracy/$name; mkdir -p $out
if [ "$name" = target ]; then
  CUDA_VISIBLE_DEVICES=$GPU SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 $COMMON --port $PORT > $out/server.log 2>&1 &
else
  GPU=$GPU PORT=$PORT bash launch_server.sh "$@" $COMMON > $out/server.log 2>&1 &
fi
server=$!
for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $server 2>/dev/null || break; sleep 5; done
curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo "$name: server failed"; exit 1; }
for c in 8 64; do
  /data/twang/sgl_eval_venv/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:$PORT/v1 --num-examples 1319 \
    --num-threads $c --temperature 0.6 --top-p 0.9 --max-tokens 1024 --out-dir $PWD/$out/gsm8k_c$c > $out/gsm8k_c$c.log 2>&1
done
python greedy_conc.py --port $PORT --n 64 --concurrency 8 --out $out/greedy.json $([ "$name" = target ] && echo --logprobs)
python dist_mixed.py --port $PORT --out $out/dist.json > $out/dist.log 2>&1
curl -s http://127.0.0.1:$PORT/server_info > $out/server_info.json
pkill -f -- "[-]-port $PORT"; sleep 15
echo "[$(date +%T)] $name done: c8 $(grep -m1 'score' $out/gsm8k_c8.log) | c64 $(grep -m1 'score' $out/gsm8k_c64.log)"
