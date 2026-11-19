"""Oracle study summary (run_oracle.sh): per (context, concurrency) the best stock EAGLE-3 config,
and how much each single fixed config loses against that per-point best (the adaptive headroom).

Metric: total throughput (tok/s). At c=1 it equals per-request speed.
python analyze_oracle.py [--root runs/oracle]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/oracle")
    args = ap.parse_args()
    data = defaultdict(dict)  # (shots, c) -> config -> summary
    for path in glob.glob(f"{args.root}/*/s*_c*.json"):
        cfg = os.path.basename(os.path.dirname(path))
        s, c = map(int, re.match(r"s(\d+)_c(\d+)\.json", os.path.basename(path)).groups())
        data[(s, c)][cfg] = json.load(open(path))["summary"]
    configs = sorted({cfg for v in data.values() for cfg in v},
                     key=lambda n: (n != "target", not n.startswith("chain"), n))
    out = {}
    for shots in sorted({s for s, _ in data}):
        points = sorted(c for s, c in data if s == shots)
        ctx = next(iter(data[(shots, points[0])].values()))["prompt_tokens"]
        print(f"\n=== context ≈ {ctx:.0f} prompt tokens ({shots}-shot) — total tok/s; accept in []")
        print(f"{'config':12s}" + "".join(f"{'c=' + str(c):>14s}" for c in points))
        for cfg in configs:
            cells = []
            for c in points:
                r = data[(shots, c)].get(cfg)
                cells.append(f"{r['total']:7.0f} [{r['accept']:.2f}]" if r else f"{'-':>14s}")
            print(f"{cfg:12s}" + "".join(f"{x:>14s}" for x in cells))
        best = {c: max(data[(shots, c)].items(), key=lambda kv: kv[1]["total"]) for c in points}
        print(f"{'ORACLE':12s}" + "".join(f"{best[c][0]:>14s}" for c in points))
        print("loss of each fixed config vs oracle (per point; worst / mean over concurrencies):")
        fixed = {}
        for cfg in configs:
            if not all(cfg in data[(shots, c)] for c in points):
                continue
            losses = [1 - data[(shots, c)][cfg]["total"] / best[c][1]["total"] for c in points]
            fixed[cfg] = (max(losses), sum(losses) / len(losses))
            print(f"  {cfg:12s}" + "".join(f"{-100 * x:>8.0f}%" for x in losses)
                  + f"   worst {-100 * max(losses):.0f}%  mean {-100 * sum(losses) / len(losses):.0f}%")
        champ = min(fixed, key=lambda k: fixed[k][1])
        print(f"best single fixed config: {champ}; adaptive (oracle) gain over it: "
              + ", ".join(f"c={c} +{100 * (best[c][1]['total'] / data[(shots, c)][champ]['total'] - 1):.0f}%"
                          for c in points))
        out[shots] = dict(ctx=ctx, best={c: best[c][0] for c in points}, champion=champ)
    json.dump(out, open(f"{args.root}/oracle_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
