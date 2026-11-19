#!/usr/bin/env bash
# Follow-up after the temperature sweep: 20k-sample switching distribution test on prompt 1
# (target vs adaptive tree), and the lossy positive control at 3000 samples x 2 prompts.
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1; export HF_HUB_OFFLINE=1
while pgrep -f "[t]emp_check.sh" > /dev/null; do sleep 30; done
sleep 20
ADAPT="--speculative-adaptive --speculative-adaptive-config configs/adaptive_tree_budget.json --speculative-eagle-topk 4 --speculative-num-steps 6"
COMMON="--mem-fraction-static 0.7 --max-running-requests 64 --return-output-ids"
run() {  # gpu port name samples prompts flags...
  local gpu=$1 port=$2 name=$3 n=$4 prompts=$5; shift 5
  local out=runs/accuracy_followup/$name; mkdir -p $out
  if [ "$name" = target ]; then
    CUDA_VISIBLE_DEVICES=$gpu SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --context-length 8192 $COMMON --port $port > $out/server.log 2>&1 &
  else
    GPU=$gpu PORT=$port bash launch_server.sh "$@" $COMMON > $out/server.log 2>&1 &
  fi
  local server=$!
  for _ in $(seq 240); do curl -sf http://127.0.0.1:$port/health > /dev/null && break; kill -0 $server 2>/dev/null || break; sleep 5; done
  curl -sf http://127.0.0.1:$port/health > /dev/null || { echo "$name: server failed"; return; }
  python dist_mixed.py --port $port --num-samples $n --prompts $prompts --out $out/dist.json > $out/dist.log 2>&1
  pkill -f -- "[-]-port $port"; sleep 15
  echo "[$(date +%T)] $name done"
}
( run 0 30000 target 20000 "0 1" ) &
( run 1 30001 adaptive_tree_budget 20000 "0 1" $ADAPT; \
  run 1 30001 lossy_adaptive_tree 3000 "0 1" $ADAPT --speculative-accept-threshold-single 0.3 --speculative-accept-threshold-acc 0.3 ) &
wait
echo "followup done"
