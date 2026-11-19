#!/usr/bin/env bash
# Controls for the 20k follow-up (prompt 1 differed at position 0, which the target's prefill samples):
# a second target-only run (test calibration) and a stock fixed EAGLE-3 tree with our tree host-plan
# defaults off (does any EAGLE server shift it?). 20k samples, prompt 1.
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1; export HF_HUB_OFFLINE=1
COMMON="--mem-fraction-static 0.7 --max-running-requests 64 --return-output-ids"
run() {  # gpu port name flags...
  local gpu=$1 port=$2 name=$3; shift 3
  local out=runs/accuracy_control/$name; mkdir -p $out
  if [ "$name" = target_rerun ]; then
    CUDA_VISIBLE_DEVICES=$gpu SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 $COMMON --port $port > $out/server.log 2>&1 &
  else
    SGLANG_OPT_FLASHINFER_VERIFY_HOST_PLAN=0 SGLANG_OPT_FLASHINFER_DRAFT_HOST_INDPTR=0 \
      GPU=$gpu PORT=$port bash launch_server.sh "$@" $COMMON > $out/server.log 2>&1 &
  fi
  local server=$!
  for _ in $(seq 240); do curl -sf http://127.0.0.1:$port/health > /dev/null && break; kill -0 $server 2>/dev/null || break; sleep 5; done
  curl -sf http://127.0.0.1:$port/health > /dev/null || { echo "$name: server failed"; return; }
  python dist_mixed.py --port $port --num-samples 20000 --prompts 1 --out $out/dist.json > $out/dist.log 2>&1
  pkill -f -- "[-]-port $port"; sleep 15
  echo "[$(date +%T)] $name done"
}
( run 0 30000 target_rerun ) &
( run 1 30001 stock_tree_n7 --speculative-eagle-topk 4 --speculative-num-steps 4 --speculative-num-draft-tokens 7 ) &
wait
echo "control done"
