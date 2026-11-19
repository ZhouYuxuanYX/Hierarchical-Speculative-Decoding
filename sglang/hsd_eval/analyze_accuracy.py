"""Accuracy / losslessness report for runs/accuracy/<cfg> (accuracy_check.sh), against target.

- GSM8K (sgl-eval, 1319 questions, T=0.6, top-p 0.9) at concurrency 8 and 64, with 1 SE.
- Greedy at concurrency 8: identical outputs, else the target's top-1 vs top-2 log-prob gap at the
  first differing position (a bf16 near-tie is benign; a large gap would be a bug).
- Distribution under state switching: first-token marginals (positions 0..7) and the (0, 1) joint,
  two-sample chi-square vs target, Bonferroni over prompts x statistics (analyze._lossless_p).
python analyze_accuracy.py
"""

import glob
import json
import math
import os

from analyze import _lossless_p

ROOT = "runs/accuracy"
ORDER = ["target", "adaptive_tree_budget", "adaptive_chain", "lossy_adaptive_tree"]


def gsm8k(cfg, c):
    found = glob.glob(f"{ROOT}/{cfg}/gsm8k_c{c}/sgl_eval_gsm8k_*/metrics.json")
    return json.load(open(found[0]))["aggregate"]["score"] if found else None


def greedy_report(ref, run):
    same, tie_gaps = 0, []
    for r, o in zip(ref, run):
        a, b = r["ids"], o["ids"]
        if a == b:
            same += 1
            continue
        pos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        tie_gaps.append(r["gaps"][pos] if pos < len(r["gaps"]) else float("nan"))
    return same, tie_gaps


def main():
    ref_greedy = json.load(open(f"{ROOT}/target/greedy.json"))
    ref_dist = {k: v for k, v in json.load(open(f"{ROOT}/target/dist.json")).items()}
    print(f"{'config':22s} {'GSM8K c=8':>10s} {'GSM8K c=64':>11s} {'greedy same':>12s} "
          f"{'max gap at diff':>16s} {'dist p (Bonf.)':>15s}")
    for cfg in ORDER:
        if not os.path.isdir(f"{ROOT}/{cfg}"):
            continue
        s8, s64 = gsm8k(cfg, 8), gsm8k(cfg, 64)
        cell = lambda s: f"{100 * s:.2f}%" if s is not None else "-"
        g = d = "-"
        if os.path.exists(f"{ROOT}/{cfg}/greedy.json") and cfg != "target":
            same, gaps = greedy_report(ref_greedy, json.load(open(f"{ROOT}/{cfg}/greedy.json")))
            g = f"{same}/{len(ref_greedy)}"
            d_gap = f"{max(gaps):.3f}" if gaps else "-"
        else:
            d_gap = "-"
        if os.path.exists(f"{ROOT}/{cfg}/dist.json") and cfg != "target":
            p, n = _lossless_p(ref_dist, json.load(open(f"{ROOT}/{cfg}/dist.json")))
            d = f"{p:.3f} ({n})"
        print(f"{cfg:22s} {cell(s8):>10s} {cell(s64):>11s} {g:>12s} {d_gap:>16s} {d:>15s}")
    se = math.sqrt(0.82 * 0.18 / 1319)
    print(f"\n1 GSM8K standard error ~ {100 * se:.2f} points; greedy gap is the target's top-1 minus "
          "top-2 log-prob at the first differing token (bf16 step ~0.06-0.125).")


if __name__ == "__main__":
    main()
