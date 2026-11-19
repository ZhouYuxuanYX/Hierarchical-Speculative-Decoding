"""Check (1) report: each spec config vs target-only greedy on the same 5 inputs.

A mismatch is benign when the target's top-2 log-prob gap at the first differing position is
tiny (a bf16 near-tie that a different batch shape can flip); a large gap would be a bug.
python compare_greedy.py runs/greedy
"""

import json
import sys

DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/greedy"


def main():
    ref = json.load(open(f"{DIR}/target_only.json"))["0.0"]
    print(f"{'config':<10} {'T':>5} {'prompt':>6}  result")
    for name in ("target_only", "chain_hsd", "tree_hsd", "tree_rrsw"):
        run = json.load(open(f"{DIR}/{name}.json"))
        for temp in ("0.0", "0.01"):
            for i, (r, o) in enumerate(zip(ref, run[temp])):
                a, b = r["ids"], o["ids"]
                n = min(len(a), len(b))
                diff = next((j for j in range(n) if a[j] != b[j]), None)
                acc = o["meta"]["spec_accept_length"]
                acc_s = f" accept_len={acc:.2f}" if acc else ""
                if diff is None and len(a) == len(b):
                    res = f"identical ({len(a)} tokens){acc_s}"
                elif diff is None:
                    res = f"same prefix, lengths {len(a)} vs {len(b)}{acc_s}"
                else:
                    gap = r["top2"][diff][0] - r["top2"][diff][1]
                    res = f"first diff at token {diff}/{n}, target top-2 logprob gap {gap:.4f}{acc_s}"
                print(f"{name:<10} {temp:>5} {i:>6}  {res}")


if __name__ == "__main__":
    main()
