"""Summarize a tree_bench.py run: per task and config, tokens/step, tokens/s, per-step ms (target forward,
verification, draft + update) and mean tokens per turn, with paired bootstrap CIs over questions (both
seeds of a question resampled together). Shadow walks (tree_bench --shadow) are excluded from the
timings and summarized separately: both verifiers' mean accepted draft tokens on the same trees."""
import collections
import glob
import json
import sys

import numpy as np

run = sys.argv[1]  # the --out directory of the tree_bench.py shards
rows = [json.loads(l) for f in sorted(glob.glob(f"{run}/shard*.jsonl")) for l in open(f)]
TASKS = ["humaneval", "gsm8k", "mt_bench", "sum", "alpaca", "qa"]  # low -> high entropy
ORDER = ["eagle", "topk", "sampled_rrsw", "sampled_hsd", "spine_rrsw", "spine_hsd", "hybrid1_rrsw", "hybrid1_hsd",
         "hybrid2_rrsw", "hybrid2_hsd", "hybrid1_hsdf", "deep2_rrsw", "deep2_hsdf", "deep4_rrsw", "deep4_hsdf"]
FULL = collections.defaultdict(lambda: 6, deep2_rrsw=8, deep2_hsdf=8, deep4_rrsw=10, deep4_hsdf=10)  # drafted tokens on the full first spine
CONFIGS = [c for c in ORDER if any(r["config"] == c for r in rows)]
PAIRS = [(a, b) for a, b in [
    ("sampled_rrsw", "topk"), ("sampled_hsd", "topk"), ("sampled_hsd", "sampled_rrsw"), ("topk", "eagle"),
    ("sampled_rrsw", "eagle"), ("sampled_hsd", "eagle"), ("spine_rrsw", "sampled_rrsw"),
    ("spine_hsd", "sampled_hsd"), ("spine_hsd", "spine_rrsw"), ("spine_rrsw", "eagle"), ("spine_hsd", "eagle"),
    ("hybrid1_hsd", "hybrid1_rrsw"), ("hybrid2_hsd", "hybrid2_rrsw"), ("hybrid1_rrsw", "sampled_rrsw"),
    ("hybrid1_hsd", "sampled_rrsw"), ("hybrid2_rrsw", "sampled_rrsw"), ("hybrid2_hsd", "sampled_rrsw")]
    if a in CONFIGS and b in CONFIGS]
# later pairs: bootstrapped with their own generator so the CIs of the pairs above do not move when they are added
NEW_PAIRS = [(a, b) for a, b in [
    ("hybrid1_hsdf", "hybrid1_rrsw"), ("deep2_rrsw", "hybrid1_rrsw"), ("deep2_hsdf", "hybrid1_hsdf"),
    ("deep2_hsdf", "deep2_rrsw"), ("deep4_rrsw", "hybrid1_rrsw"), ("deep4_hsdf", "hybrid1_hsdf"),
    ("deep4_hsdf", "deep4_rrsw"), ("hybrid1_rrsw", "eagle"), ("hybrid1_hsdf", "eagle"), ("deep2_rrsw", "eagle"),
    ("deep2_hsdf", "eagle"), ("deep4_rrsw", "eagle"), ("deep4_hsdf", "eagle")]
    if a in CONFIGS and b in CONFIGS]
# per (task, question) -> config -> summed [new tokens, steps, wall, target fwd, verify, update, turns]
agg = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(7)))
# per (task, question) -> config -> summed [steps, shadow rrsw accepted, shadow hsd accepted, spine, full spine]
shadow = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(5)))
for r in rows:
    s = r.get("shadow_time", 0.0)
    agg[(r["task"], r["question_id"])][r["config"]] += np.array(
        [r["new_tokens"], r["steps"], r["wall"] - s, r["target_fwd"], r["verify"] - s, r["update"], 1])
    if "spine" in r:
        sp = np.array(r["spine"])
        shadow[(r["task"], r["question_id"])][r["config"]] += np.array(
            [len(sp), sum(r.get("shadow_rrsw", [0])), sum(r.get("shadow_hsd", [0])), sp.sum(),
             (sp == FULL[r["config"]]).sum()])
print(f"{len(rows)} generated turns from {len(glob.glob(run + '/shard*.jsonl'))} shards; configs {CONFIGS}")
rng = np.random.default_rng(0)
rng_new = np.random.default_rng(1)
q = lambda v: f"{100 * np.mean(v):+.1f}% [{100 * np.quantile(v, .025):+.1f}, {100 * np.quantile(v, .975):+.1f}]"


def summarize(keys, label):
    keys = [k for k in keys if all(c in agg[k] for c in CONFIGS)]
    if not keys:
        return
    M = {c: np.array([agg[k][c] for k in keys]) for c in CONFIGS}
    print(f"{label} (n={len(keys)} questions)")
    for c in CONFIGS:
        t = M[c].sum(0)
        print(f"  {c:<13} {t[0] / t[1]:.3f} tok/step {t[0] / t[2]:6.1f} tok/s | ms/step: target {1e3 * t[3] / t[1]:.2f} "
              f"verify {1e3 * t[4] / t[1]:.2f} draft+update {1e3 * t[5] / t[1]:.2f} | {t[0] / t[6]:.0f} tok/turn")
    n = len(keys)
    for a, b in PAIRS + NEW_PAIRS:
        be, ds, gen = [], [], rng if (a, b) in PAIRS else rng_new
        for _ in range(2000):
            i = gen.integers(0, n, n)
            A, B = M[a][i].sum(0), M[b][i].sum(0)
            be.append((A[0] / A[1]) / (B[0] / B[1]) - 1)
            ds.append((A[0] / A[2]) / (B[0] / B[2]) - 1)
        print(f"    {a} vs {b}: tok/step {q(be)}  tok/s {q(ds)}")
    for c in CONFIGS:
        S = np.array([shadow[k][c] for k in keys])
        if S[:, 0].sum() == 0:
            continue
        t = S.sum(0)
        line = (f"    {c}: root first-child path {t[3] / t[0]:.2f} tokens, full length ({FULL[c]}) on "
                f"{t[4] / t[0]:.0%} of steps")
        if S[:, 2].any():
            d = [(lambda T: T[2] / T[1] - 1)(S[rng.integers(0, n, n)].sum(0)) for _ in range(2000)]
            line += f"; same trees: accepted/step rrsw {t[1] / t[0]:.3f} hsd {t[2] / t[0]:.3f}, hsd vs rrsw {q(d)}"
        print(line)


for task in TASKS:
    summarize([k for k in agg if k[0] == task], task)
summarize(list(agg), "all")
