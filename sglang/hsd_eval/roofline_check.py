"""Ideal (roofline) vs measured step time for the stock EAGLE-3 oracle runs.

Ideal step = draft + verify, each pass max(memory time, compute time):
  draft : (d-1) decode passes over B*k tokens + 1 draft-extend pass over B*N tokens,
          head = 0.42B params (0.85 GB, 0.85 GFLOP/token)
  verify: target 8.03B (16.06 GB weights; ~15 GFLOP/token excluding the embedding table)
          + KV read B * ctx * 128 KiB (32 layers x 8 KV heads x 128 dims x K,V x bf16)
Measured step = sum(request latency) / sum(verify steps), from oracle_client rows.
"""
import json
BW, FLOPS = 936e9, 120e12                    # measured ceilings (hw_peaks.py)
W_T, F_T, W_D, F_D, KV = 16.06e9, 15e9, 0.85e9, 0.85e9, 131072
OUT = 170                                    # mean GSM8K answer length
SHAPES = {"target": (0, 1, 1), "chain3": (3, 1, 4), "chain6": (6, 1, 7), "tree4x6n7": (6, 4, 7),
          "tree4x6n16": (6, 4, 16), "tree8x5n32": (5, 8, 32), "tree10x5n60": (5, 10, 60)}
def ideal(B, d, k, N, ctx):
    t_mem_v = (W_T + B * ctx * KV) / BW
    t_cmp_v = B * N * F_T / FLOPS
    draft = 0.0 if d == 0 else (d - 1) * max(W_D / BW, B * k * F_D / FLOPS) + max(W_D / BW, B * N * F_D / FLOPS)
    return draft + max(t_mem_v, t_cmp_v), ("compute" if t_cmp_v > t_mem_v else "memory"), draft, max(t_mem_v, t_cmp_v)
rows = []
for shots, ctx0 in ((0, 115), (8, 1846)):
    for name, (d, k, N) in SHAPES.items():
        for c in (1, 2, 4, 8, 16, 32, 64):
            r = json.load(open(f"runs/oracle/{name}/s{shots}_c{c}.json"))
            reqs = r["rows"]
            t_meas = sum(x["latency"] for x in reqs) / sum(x["steps"] for x in reqs)
            A = r["summary"]["accept"]
            t_id, regime, t_draft, t_ver = ideal(c, d, k, N, ctx0 + OUT / 2)
            rows.append(dict(ctx=ctx0, cfg=name, c=c, BN=c * N, regime=regime, A=A,
                             t_ideal=t_id * 1e3, t_draft=t_draft * 1e3, t_verify=t_ver * 1e3,
                             t_meas=t_meas * 1e3, eff=t_id / t_meas,
                             thr_ideal=c * A / t_id, thr_meas=c * A / t_meas))
json.dump(rows, open("runs/oracle/roofline_check.json", "w"), indent=1)
for ctx in (115, 1846):
    print(f"\n=== context {ctx} ===  step ms: ideal (draft+verify) | measured | efficiency ; [regime of verify]")
    print("config        " + "".join(f"{'c=' + str(c):>22s}" for c in (1, 4, 16, 64)))
    for name in SHAPES:
        cells = []
        for c in (1, 4, 16, 64):
            x = next(r for r in rows if r["ctx"] == ctx and r["cfg"] == name and r["c"] == c)
            cells.append(f"{x['t_ideal']:5.1f}|{x['t_meas']:5.1f} {100*x['eff']:3.0f}% {x['regime'][0]}")
        print(f"{name:12s}  " + "".join(f"{s:>22s}" for s in cells))
