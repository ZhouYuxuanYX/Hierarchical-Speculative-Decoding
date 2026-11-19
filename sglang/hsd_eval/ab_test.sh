#!/usr/bin/env bash
# A/B of a server-side env flag on one config: greedy exactness (c=1) + speed (c=1, c=4).
# Usage: GPU=0 PORT=30000 FLAG=SGLANG_OPT_FLASHINFER_VERIFY_HOST_PLAN bash ab_test.sh <config> <tag>
set -uo pipefail
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1; export HF_HUB_OFFLINE=1
GPU=${GPU:-0}; PORT=${PORT:-30000}; name=$1; tag=$2
source <(sed -n '/^flags_of()/,/^}/p' run_oracle.sh); COMMON="--mem-fraction-static 0.7 --max-running-requests 64"
for val in 0 1; do
  out=runs/ab/$tag/$name/$FLAG=$val; mkdir -p $out
  env $FLAG=$val GPU=$GPU PORT=$PORT bash launch_server.sh $(flags_of $name) --return-output-ids > $out/server.log 2>&1 &
  server=$!
  for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $server 2>/dev/null || break; sleep 5; done
  curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo "$val: server failed"; continue; }
  [ -n "${SKIP_GREEDY:-}" ] || python greedy_dump.py --port $PORT --out $out/greedy.json
  for rep in 1 2; do
    for s in ${SHOTS:-0}; do
      for c in ${CONCS:-1 4}; do
        n=$(( c == 1 ? 24 : (c == 4 ? 100 : 3 * c) ))
        python oracle_client.py --port $PORT --shots $s --concurrency $c --num-requests $n --out $out/c${c}_s${s}_r$rep.json > /dev/null
      done
    done
  done
  pkill -f -- "[-]-port $PORT"; sleep 15
done
python - <<PY
import json, glob, statistics as st
base = "runs/ab/$tag/$name/$FLAG"
import os
if os.path.exists(f"{base}=0/greedy.json"):
    g0, g1 = (json.load(open(f"{base}={v}/greedy.json")) for v in (0, 1))
    print(f"greedy identical: {sum(a == b for a, b in zip(g0, g1))}/{len(g0)}")
for s in "${SHOTS:-0}".split():
  for c in [int(x) for x in "${CONCS:-1 4}".split()]:
    for v in (0, 1):
        runs = [json.load(open(f))["summary"] for f in sorted(glob.glob(f"{base}={v}/c{c}_s{s}_r*.json"))]
        print(f"shots={s} c={c} $FLAG={v}: total tok/s {[round(r['total']) for r in runs]} mean {st.mean(r['total'] for r in runs):.0f}; accept {st.mean(r['accept'] for r in runs):.2f}")
PY
