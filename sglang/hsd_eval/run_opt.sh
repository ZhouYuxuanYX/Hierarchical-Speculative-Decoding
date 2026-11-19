#!/usr/bin/env bash
# Draft-shape sweep for HSD on GSM8K: which (topk, depth) is fastest at each concurrency?
# One fresh server per config; acceptance is read after the c=1 run (concurrency does not change it),
# then the same server serves c=4, 16, 64. More questions at higher concurrency for a steady state.
# Usage: GPU=0 PORT=30000 bash run_opt.sh <config>...   config = target | chain<d> | tree<k>x<d>   (HSD)
#        | stock_chain<d> | stock<k>x<d>n<N>   (stock SGLang EAGLE-3: score-pruned tree of N tokens, stock
#          verification, draft CUDA graphs; OUT=runs/opt_stock to keep them apart)
set -uo pipefail
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1
export HF_HUB_OFFLINE=1
GPU=${GPU:-0}; PORT=${PORT:-30000}; OUT=${OUT:-runs/opt}
declare -A NQ=([1]=100 [4]=100 [16]=200 [64]=400)

flags_of() {  # unpruned sampled tree: draft tokens = 1 + k + (d-1) k^2 (chain: k=1 -> d+1)
  local name=$1 k d n run=64
  case $name in
    target) echo "NOSPEC"; return ;;
    stock_chain*) d=${name#stock_chain}
      echo "--speculative-num-steps $d --speculative-eagle-topk 1 --speculative-num-draft-tokens $((d + 1)) \
--mem-fraction-static 0.6 --max-running-requests 64"; return ;;
    stock*) k=${name#stock}; k=${k%%x*}; d=${name#*x}; d=${d%n*}; n=${name#*n}
      echo "--speculative-num-steps $d --speculative-eagle-topk $k --speculative-num-draft-tokens $n \
--mem-fraction-static 0.6 --max-running-requests 64"; return ;;
    chain*) k=1; d=${name#chain} ;;
    tree*)  k=${name#tree}; k=${k%x*}; d=${name#*x} ;;
  esac
  n=$((1 + k + (d - 1) * k * k))
  [ $n -ge 49 ] && run=32  # per-node fp32 prob rows: big trees OOM above ~32 running requests
  echo "--speculative-num-steps $d --speculative-eagle-topk $k --speculative-num-draft-tokens $n \
--speculative-use-hsd --mem-fraction-static 0.6 --max-running-requests $run"
}

for name in "$@"; do
  if curl -sf http://127.0.0.1:$PORT/health > /dev/null; then echo "port $PORT busy, abort"; exit 1; fi
  mkdir -p $OUT/$name
  flags=$(flags_of $name); echo "$flags" > $OUT/$name/flags.txt
  if [ "$flags" = "NOSPEC" ]; then
    CUDA_VISIBLE_DEVICES=$GPU SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 \
      --mem-fraction-static 0.6 --max-running-requests 64 --port $PORT > $OUT/$name/server.log 2>&1 &
  else
    GPU=$GPU PORT=$PORT bash launch_server.sh $flags > $OUT/$name/server.log 2>&1 &
  fi
  for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
  for c in 1 4 16 64; do
    out=$OUT/$name/c$c; mkdir -p $out
    /data/twang/sgl_eval_venv/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:$PORT/v1 \
      --num-examples ${NQ[$c]} --num-threads $c --temperature 0.6 --top-p 0.9 --max-tokens 1024 \
      --out-dir $PWD/$out > $out/gsm8k.log 2>&1
    [ $c = 1 ] && curl -s http://127.0.0.1:$PORT/server_info > $OUT/$name/server_info_c1.json
  done
  pkill -f -- "[-]-port $PORT"; sleep 15
  echo "[$(date +%T)] $name done"
done
