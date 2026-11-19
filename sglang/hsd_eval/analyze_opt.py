"""Summarize the draft-shape sweep (run_opt.sh): acceptance, accuracy, per-request and total throughput
per config and concurrency, the measured best config per concurrency, and the roofline prediction.

python analyze_opt.py [--root runs/opt]
"""

import argparse
import glob
import json
import os

import numpy as np

# One Llama-3.1-8B bf16 forward pass on this GPU (RTX 6000 Ada), N new tokens -> ms (fwd_cost.py).
FWD_N = [1, 7, 16, 32, 64, 85, 128, 170, 224, 256, 384, 512, 680, 1024, 1360, 2048, 2720]
FWD_MS = [18.98, 20.30, 20.66, 22.23, 22.34, 23.03, 24.42, 29.58, 36.00, 42.40, 61.26, 75.10,
          108.07, 151.73, 204.13, 310.94, 437.94]
CONC = [1, 4, 16, 64]


def fwd_ms(n):
    return float(np.interp(n, FWD_N, FWD_MS)) if n <= FWD_N[-1] else FWD_MS[-1] + (n - FWD_N[-1]) * 0.169


def shape(name):
    if name == "target":
        return 1, 0, 1
    if name.startswith("stock_chain"):
        d = int(name[11:])
        return 1, d, d + 1
    if name.startswith("stock"):  # stock<k>x<d>n<N>: score-pruned tree of N tokens
        k, rest = name[5:].split("x")
        d, n = rest.split("n")
        return int(k), int(d), int(n)
    if name.startswith("chain"):
        k, d = 1, int(name[5:])
    else:
        k, d = map(int, name[4:].split("x"))
    return k, d, 1 + k + (d - 1) * k * k


def run_stats(path):
    files = glob.glob(f"{path}/sgl_eval*/output-rs0.jsonl")
    if not files or not glob.glob(f"{path}/sgl_eval*/metrics.json"):  # missing or still running
        return None
    rows = [json.loads(line) for line in open(files[0])]
    m = json.load(open(glob.glob(f"{path}/sgl_eval*/metrics.json")[0]))
    tok = sum(r["num_generated_tokens"] for r in rows)
    lat = sum(r["generation_end_time"] - r["generation_start_time"] for r in rows)
    return dict(n=len(rows), acc=sum(r["symbolic_correct"] for r in rows) / len(rows),
                per_req=tok / lat, total=m["total_completion_tokens"] / m["latency_seconds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/opt")
    args = ap.parse_args()
    res = {}
    for d in sorted(glob.glob(f"{args.root}/*/")):
        name = os.path.basename(d.rstrip("/"))
        k, depth, nodes = shape(name)
        info = f"{d}/server_info_c1.json"
        accept = 1.0 if name == "target" else float("nan")  # nan until the c=1 run has finished
        if name != "target" and os.path.exists(info):
            si = json.load(open(info))
            accept = si.get("internal_states", [si])[0].get("avg_spec_accept_length", float("nan"))
        res[name] = dict(k=k, depth=depth, nodes=nodes, accept=accept,
                         runs={c: run_stats(f"{d}/c{c}") for c in CONC})
    order = sorted(res, key=lambda n: (res[n]["k"] > 1, res[n]["k"], res[n]["nodes"]))
    print("config     k depth nodes accept |" + "".join(f"  c={c}: per-req / total" for c in CONC))
    for n in order:
        r = res[n]
        cells = "".join(
            f"  {s['per_req']:7.1f} / {s['total']:6.0f}    " if s else "        -            "
            for s in (r["runs"][c] for c in CONC)
        )
        print(f"{n:10s} {r['k']} {r['depth']:5d} {r['nodes']:5d} {r['accept']:6.2f} |{cells}")
    print("\nbest config per concurrency (by total throughput; per-request in parentheses)")
    for c in CONC:
        done = [(n, r["runs"][c]) for n, r in res.items() if r["runs"][c]]
        if not done:
            continue
        best = max(done, key=lambda x: x[1]["total"])
        # roofline prediction: B running requests, target forward only (draft cost ignored)
        pred = max(res, key=lambda n: min(c, (64 if res[n]["nodes"] < 49 or n.startswith("stock") else 32)) * res[n]["accept"]
                   / fwd_ms(min(c, (64 if res[n]["nodes"] < 49 or n.startswith("stock") else 32)) * res[n]["nodes"]))
        print(f"  c={c:2d}: measured best {best[0]:9s} {best[1]['total']:6.0f} tok/s "
              f"({best[1]['per_req']:.1f}) | roofline pick {pred}")
    json.dump(res, open(f"{args.root}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
