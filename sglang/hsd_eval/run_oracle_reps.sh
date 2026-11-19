#!/usr/bin/env bash
# 3 repeats of the oracle grid: GPU 0 and GPU 1 each take half the configs per repeat.
cd "$(dirname "$0")"
A="adaptive_tree_budget chain3 tree4x6n16 tree10x5n60 target"
B="adaptive_chain chain6 tree4x6n7 tree8x5n32"
for r in 1 2 3; do
  OUT=runs/oracle_r$r SHOTS="0 8" GPU=0 PORT=30000 bash run_oracle.sh $A > runs/oracle_r${r}_gpu0.log 2>&1 &
  OUT=runs/oracle_r$r SHOTS="0 8" GPU=1 PORT=30001 bash run_oracle.sh $B > runs/oracle_r${r}_gpu1.log 2>&1 &
  wait
done
echo "all repeats done"
