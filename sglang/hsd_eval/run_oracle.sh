#!/usr/bin/env bash
# Oracle study, stock SGLang EAGLE-3: best draft shape per (concurrency, context length).
# One fresh server per config; for each context (K-shot prefix) and concurrency, oracle_client.py
# sends max(24, 3c) GSM8K requests (T=0.6, top-p 0.9) and records tokens, verify steps, latency.
# Usage: GPU=0 PORT=30000 bash run_oracle.sh <config>...
#   config = target | chain<d> | tree<k>x<d>n<N>   (stock EAGLE-3, score-pruned tree of N tokens)
#          | adaptive_chain (SGLang --speculative-adaptive, default policy) | adaptive_<cfg> (configs/<cfg>.json)
set -uo pipefail
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1
export HF_HUB_OFFLINE=1
GPU=${GPU:-0}; PORT=${PORT:-30000}; OUT=${OUT:-runs/oracle}
SHOTS=${SHOTS:-"0 6 20"}; CONCS=${CONCS:-"1 2 4 8 16 32 64"}
COMMON="--mem-fraction-static 0.7 --max-running-requests 64"

flags_of() {
  local name=$1 k d n
  case $name in
    target) echo "NOSPEC" ;;
    adaptive_chain) echo "--speculative-adaptive --speculative-eagle-topk 1 $COMMON" ;;  # SGLang default policy
    adaptive_tree*) echo "--speculative-adaptive --speculative-adaptive-config configs/${name}.json \
--speculative-eagle-topk 4 --speculative-num-steps 6 $COMMON" ;;  # e.g. adaptive_tree_k4
    chain*) d=${name#chain}
      echo "--speculative-num-steps $d --speculative-eagle-topk 1 --speculative-num-draft-tokens $((d + 1)) $COMMON" ;;
    tree*) k=${name#tree}; k=${k%%x*}; d=${name#*x}; d=${d%n*}; n=${name#*n}
      echo "--speculative-num-steps $d --speculative-eagle-topk $k --speculative-num-draft-tokens $n $COMMON" ;;
  esac
}

for name in "$@"; do
  if curl -sf http://127.0.0.1:$PORT/health > /dev/null; then echo "port $PORT busy, abort"; exit 1; fi
  mkdir -p $OUT/$name
  flags=$(flags_of $name); echo "$flags" > $OUT/$name/flags.txt
  if [ "$flags" = "NOSPEC" ]; then
    CUDA_VISIBLE_DEVICES=$GPU SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 \
      $COMMON --port $PORT > $OUT/$name/server.log 2>&1 &
  else
    GPU=$GPU PORT=$PORT bash launch_server.sh $flags > $OUT/$name/server.log 2>&1 &
  fi
  server=$!
  for _ in $(seq 240); do
    curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
    kill -0 $server 2> /dev/null || break  # server exited during startup
    sleep 5
  done
  if ! curl -sf http://127.0.0.1:$PORT/health > /dev/null; then
    echo "[$(date +%T)] $name: server failed to start, see $OUT/$name/server.log"; continue
  fi
  for s in $SHOTS; do
    for c in $CONCS; do
      n=$(( 3 * c > 24 ? 3 * c : 24 ))
      python oracle_client.py --port $PORT --shots $s --concurrency $c --num-requests $n \
        --out $OUT/$name/s${s}_c${c}.json > /dev/null 2>> $OUT/$name/client.err \
        || echo "[$(date +%T)] $name s=$s c=$c FAILED"
    done
  done
  pkill -f -- "[-]-port $PORT"; sleep 15
  echo "[$(date +%T)] $name done"
done
