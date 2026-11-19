"""Summarize runs/: GSM8K score, accept length, and losslessness against target_only.

Losslessness: for every prompt, the first-token marginals at positions 0..7 and the
joint of positions (0, 1) are compared with the target_only samples by a two-sample
chi-square test (categories under 10 combined samples are pooled). The reported p is
the smallest over prompts and statistics, Bonferroni-corrected for their number.
"""

import glob
import json
import os
import sys
from collections import Counter

from scipy.stats import chi2_contingency

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
ORDER = ["target_only", "chain_token", "chain_hsd", "tree_rrsw", "tree_hsd", "lossy_ctrl"]


def _pvalue(a, b, min_count=10):
    ca, cb = Counter(a), Counter(b)
    keys = [k for k in set(ca) | set(cb) if ca[k] + cb[k] >= min_count]
    rest_a = len(a) - sum(ca[k] for k in keys)
    rest_b = len(b) - sum(cb[k] for k in keys)
    table = [[ca[k] for k in keys] + [rest_a], [cb[k] for k in keys] + [rest_b]]
    table = [list(col) for col in zip(*table) if sum(col) > 0]
    if len(table) < 2:
        return 1.0
    return chi2_contingency(list(zip(*table)))[1]


def _lossless_p(ref, run):
    pvals = []
    for prompt, ref_samples in ref.items():
        samples = run[prompt]
        for pos in range(min(len(ref_samples[0]), len(samples[0]))):
            pvals.append(_pvalue([s[pos] for s in ref_samples], [s[pos] for s in samples]))
        pvals.append(
            _pvalue([tuple(s[:2]) for s in ref_samples], [tuple(s[:2]) for s in samples])
        )
    return min(1.0, min(pvals) * len(pvals)), len(pvals)


def _gsm8k(run_dir):
    found = glob.glob(f"{run_dir}/sgl_eval_gsm8k_*/metrics.json")
    if not found:
        return None
    return json.load(open(sorted(found)[-1]))["aggregate"]["score"]


def main():
    ref_path = f"{RUNS}/target_only/dist.json"
    ref = json.load(open(ref_path)) if os.path.exists(ref_path) else None
    print(f"{'config':<12} {'gsm8k':>7} {'accept_len':>10} {'lossless p (Bonf.)':>19}")
    for name in ORDER:
        run_dir = f"{RUNS}/{name}"
        if not os.path.isdir(run_dir):
            continue
        score = _gsm8k(run_dir)
        info_path = f"{run_dir}/server_info_after_gsm8k.json"
        accept = "-"
        if os.path.exists(info_path) and name != "target_only":
            accept = f"{json.load(open(info_path))['internal_states'][0]['avg_spec_accept_length']:.3f}"
        lossless = "-"
        dist_path = f"{run_dir}/dist.json"
        if ref is not None and name != "target_only" and os.path.exists(dist_path):
            p, n = _lossless_p(ref, json.load(open(dist_path)))
            lossless = f"{p:.3g} ({n} tests)"
        score_s = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"{name:<12} {score_s:>7} {accept:>10} {lossless:>19}")


if __name__ == "__main__":
    main()
