#!/usr/bin/env bash
# Start the temperature sweep once the current accuracy checks have freed both GPUs.
cd "$(dirname "$0")"
while pgrep -f "[a]ccuracy_check.sh" > /dev/null; do sleep 30; done
ADAPT="--speculative-adaptive --speculative-adaptive-config configs/adaptive_tree_budget.json --speculative-eagle-topk 4 --speculative-num-steps 6"
GPU=0 PORT=30000 bash temp_check.sh target > runs/accuracy_temp_gpu0.log 2>&1 &
GPU=1 PORT=30001 bash temp_check.sh adaptive_tree_budget $ADAPT > runs/accuracy_temp_gpu1.log 2>&1 &
wait
