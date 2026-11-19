"""Per-decode-step breakdown of a torch-profiler trace: wall time, GPU busy (union of kernels), idle,
and GPU time by kernel family. Steps = profiled forward iterations (num_steps passed to /start_profile).
python trace_breakdown.py runs/profile/<cfg>/c<c> [num_steps]
"""
import glob, gzip, json, sys
from collections import defaultdict
path = glob.glob(f"{sys.argv[1]}/*.trace.json.gz")[0]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ev = json.load(gzip.open(path))["traceEvents"]
k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
k.sort(key=lambda e: e["ts"])
t0, t1 = k[0]["ts"], max(e["ts"] + e["dur"] for e in k)
busy, cur_s, cur_e = 0.0, None, None
for e in k:
    s, f = e["ts"], e["ts"] + e["dur"]
    if cur_e is None or s > cur_e:
        if cur_e is not None: busy += cur_e - cur_s
        cur_s, cur_e = s, f
    else:
        cur_e = max(cur_e, f)
busy += cur_e - cur_s
def family(n):
    n = n.lower()
    if "gemv" in n: return "gemv"
    if "gemm" in n or "cutlass" in n and "rmsnorm" not in n: return "gemm"
    if "flashinfer" in n and ("prefill" in n or "decode" in n or "attention" in n): return "attention"
    if "rmsnorm" in n or "norm" in n: return "norm"
    if "topk" in n or "sort" in n or "radix" in n: return "topk/sort"
    if "softmax" in n or "sampling" in n or "sample" in n: return "softmax/sample"
    if "tree" in n or "verify" in n or "eagle" in n or "build" in n: return "tree/verify kernels"
    if "memcpy" in n or "memset" in n or e.get("cat") != "kernel": return "memcpy/memset"
    return "other elementwise"
fam = defaultdict(float); cnt = defaultdict(int)
for e in k:
    f = family(e["name"]); fam[f] += e["dur"]; cnt[f] += 1
wall = t1 - t0
print(f"{sys.argv[1]}: window {wall/1e3:.1f} ms over {steps} steps -> {wall/steps/1e3:.2f} ms/step; "
      f"GPU busy {busy/steps/1e3:.2f} ms/step ({100*busy/wall:.0f}%), idle {(wall-busy)/steps/1e3:.2f} ms/step; "
      f"{len(k)/steps:.0f} kernels/step")
for f, d in sorted(fam.items(), key=lambda x: -x[1]):
    print(f"   {f:22s} {d/steps/1e3:7.2f} ms/step  {cnt[f]/steps:6.0f} launches/step")
