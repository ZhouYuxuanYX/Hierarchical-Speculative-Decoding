#!/usr/bin/env bash
# Batch-1 speed in SGLang on GPU 0, one fresh server per config.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
out=runs/bs1; mkdir -p $out
CHAIN="--speculative-num-steps 6 --speculative-eagle-topk 1 --speculative-num-draft-tokens 7"
TREE="--speculative-num-steps 6 --speculative-eagle-topk 4 --speculative-num-draft-tokens 85 --mem-fraction-static 0.6 --max-running-requests 32"
wait_up() { for _ in $(seq 240); do curl -sf http://127.0.0.1:30000/health > /dev/null && return 0; sleep 5; done; return 1; }
run() {
  local name=$1; shift
  # A leftover server on the port would silently answer for this config.
  if curl -sf http://127.0.0.1:30000/health > /dev/null; then echo "$name: port 30000 still busy, abort"; exit 1; fi
  "$@" > $out/server_$name.log 2>&1 &
  wait_up && echo "$name $(python bs1_bench.py --port 30000 --out $out/$name.json 2>/dev/null | tail -1)"
  pkill -f -- "[-]-port 30000"; sleep 15
}
# target_only: valid from the first run (58.6 tok/s), not re-run
run chain_token env GPU=0 PORT=30000 bash launch_server.sh $CHAIN --speculative-use-rejection-sampling
run chain_hsd   env GPU=0 PORT=30000 bash launch_server.sh $CHAIN --speculative-use-hsd
run tree_rrsw   env GPU=0 PORT=30000 bash launch_server.sh $TREE --speculative-use-rejection-sampling
run tree_hsd    env GPU=0 PORT=30000 bash launch_server.sh $TREE --speculative-use-hsd
# SGLang's stock EAGLE tree (top-k children, pruned to 60 tokens, token-wise target-only verify)
run eagle_tree  env GPU=0 PORT=30000 bash launch_server.sh --speculative-num-steps 5 --speculative-eagle-topk 10 --speculative-num-draft-tokens 60
echo "bs1 all done"
