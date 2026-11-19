"""Repeated oracle grid (run_oracle_reps.sh -> runs/oracle_r{1,2,3}): mean total tok/s and spread
per config/point, the oracle (best fixed config by mean), and each adaptive mode's loss against it.

python analyze_oracle_reps.py
"""

import glob
import json
import os
import statistics as st
from collections import defaultdict

CONC = [1, 2, 4, 8, 16, 32, 64]
REPS = sorted(glob.glob("runs/oracle_r*/"))


def main():
    vals = defaultdict(list)  # (cfg, shots, c) -> [total tok/s per repeat]
    vpp = defaultdict(list)  # verified tokens per pass
    for rep in REPS:
        for path in glob.glob(f"{rep}*/s*_c*.json"):
            cfg = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)[:-5]
            shots, c = int(name.split("_")[0][1:]), int(name.split("_c")[1])
            s = json.load(open(path))["summary"]
            vals[(cfg, shots, c)].append(s["total"])
            if s.get("verify_passes"):
                vpp[(cfg, shots, c)].append(s["verify_tokens_per_pass"])
    cfgs = sorted({k[0] for k in vals}, key=lambda n: (n.startswith("adaptive"), n))
    fixed = [c for c in cfgs if not c.startswith("adaptive")]
    out = {}
    for shots in sorted({k[1] for k in vals}):
        print(f"\n=== {shots}-shot context — total tok/s, mean of {len(REPS)} repeats (± range/2)")
        print(f"{'config':22s}" + "".join(f"{'c=' + str(c):>14s}" for c in CONC))
        mean = {}
        for cfg in cfgs:
            row = []
            for c in CONC:
                v = vals.get((cfg, shots, c), [])
                mean[(cfg, c)] = st.mean(v) if v else float("nan")
                row.append(f"{st.mean(v):7.0f}±{(max(v) - min(v)) / 2:<4.0f}" if v else f"{'-':>12s}")
            print(f"{cfg:22s}" + "".join(f"{x:>14s}" for x in row))
        best = {c: max(fixed, key=lambda f: mean[(f, c)]) for c in CONC}
        print(f"{'ORACLE (fixed)':22s}" + "".join(f"{best[c]:>14s}" for c in CONC))
        for cfg in cfgs:
            if not cfg.startswith("adaptive"):
                continue
            rel = [mean[(cfg, c)] / mean[(best[c], c)] - 1 for c in CONC]
            print(f"{cfg + ' vs oracle':22s}" + "".join(f"{100 * r:>13.0f}%" for r in rel)
                  + f"   mean {100 * st.mean(rel):.0f}%")
            if (cfg, shots, 1) in vpp:
                print(f"{'  verify tok/pass':22s}" + "".join(
                    f"{st.mean(vpp[(cfg, shots, c)]):>14.0f}" if vpp.get((cfg, shots, c)) else f"{'-':>14s}"
                    for c in CONC))
        out[shots] = {"best": best, "mean": {f"{k[0]}@{k[1]}": v for k, v in mean.items()}}
    json.dump(out, open("runs/oracle_reps_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
