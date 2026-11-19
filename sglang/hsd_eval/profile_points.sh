#!/usr/bin/env bash
# Torch-profiler traces of stock EAGLE-3 decode steps under steady load.
# Usage: GPU=0 PORT=30000 bash profile_points.sh <config> <concurrency>...
set -uo pipefail
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1
export HF_HUB_OFFLINE=1 SGLANG_ADMIN_API_KEY=sk-admin
GPU=${GPU:-0}; PORT=${PORT:-30000}; name=$1; shift
source <(sed -n '/^flags_of()/,/^}/p' run_oracle.sh); COMMON="--mem-fraction-static 0.7 --max-running-requests 64"
out=runs/profile/${TAG:-}$name; mkdir -p $out
GPU=$GPU PORT=$PORT bash launch_server.sh $(flags_of $name) --admin-api-key sk-admin > $out/server.log 2>&1 &
server=$!
for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $server 2>/dev/null || break; sleep 5; done
curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo "$name: server failed"; exit 1; }
for c in "$@"; do
  # load that outlasts the capture: 6c requests at concurrency c
  python oracle_client.py --port $PORT --shots 0 --concurrency $c --num-requests $((20 * c > 40 ? 20 * c : 40)) \
    --out $out/load_c$c.json > /dev/null 2>&1 &
  load=$!
  sleep 20  # warm-up and steady state
  python - <<PY
import requests
r = requests.post("http://127.0.0.1:$PORT/start_profile", headers={"Authorization": "Bearer sk-admin"},
    json={"output_dir": "$PWD/$out/c$c", "num_steps": "30", "activities": ["CPU", "GPU"]})
print("$name c=$c profile:", r.status_code, r.text[:200])
PY
  wait $load
done
pkill -f -- "[-]-port $PORT"; sleep 10
