#!/usr/bin/env bash
# Check (2) on GPU 1, after run_greedy.sh. Server and kernel come from the debug worktree
# /data/twang/sglang_debug (= SGLang-tree-hsd + an uncommitted root-q dump hook).
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
until grep -q "greedy all done" runs/greedy_run.log; do sleep 20; done
out=runs/exact; dump=/data/twang/hsd_root_q_dump; mkdir -p $out $dump
export PYTHONPATH=/data/twang/sglang_debug/python
SGLANG_DEBUG_DUMP_ROOT_Q=$dump GPU=1 PORT=30001 bash launch_server.sh \
  --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --speculative-use-hsd \
  > $out/server.log 2>&1 &
for _ in $(seq 240); do curl -sf http://127.0.0.1:30001/health > /dev/null && break; sleep 5; done
CUDA_VISIBLE_DEVICES=1 python exact_real.py --port 30001 --dump-dir $dump --out $out/report.json 2>&1 | grep -v "^DEBUG" > $out/exact.log
pkill -f -- "[-]-port 30001 "
echo "exact done"
