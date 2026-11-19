"""Batch-1 speed tables: the reference code (runs/ref/bench.jsonl) and SGLang (runs/bs1/*.json)."""
import glob
import json
import os
from collections import defaultdict

ref = defaultdict(lambda: [0, 0.0, 0])  # tokens, wall, steps
if os.path.exists("runs/ref/bench.jsonl"):
    for line in open("runs/ref/bench.jsonl"):
        r = json.loads(line)
        a = ref[r["config"]]
        a[0] += r["new_tokens"]; a[1] += r["wall"]; a[2] += r["steps"]
    base = ref["chain_hsd"][0] / ref["chain_hsd"][1] if ref["chain_hsd"][1] else None
    print("Reference code (paper's EAGLE-3 implementation, HF transformers), batch 1, T=1")
    print(f"{'config':<14} {'turns':>5} {'tok/step':>8} {'tok/s':>7} {'vs chain_hsd':>12}")
    turns = defaultdict(int)
    for line in open("runs/ref/bench.jsonl"):
        turns[json.loads(line)["config"]] += 1
    for c, (t, w, s) in ref.items():
        print(f"{c:<14} {turns[c]:>5} {t / s:>8.3f} {t / w:>7.1f} {(t / w) / base if base else float('nan'):>11.2f}x")
print()
print("SGLang (our branch), batch 1, T=1")
rows = {os.path.basename(p)[:-5]: json.load(open(p))["summary"] for p in glob.glob("runs/bs1/*.json")}
base = rows.get("chain_hsd", {}).get("tok_per_s")
print(f"{'config':<14} {'n':>5} {'tok/step':>8} {'tok/s':>7} {'vs chain_hsd':>12}")
for c in ["target_only", "chain_token", "chain_hsd", "tree_rrsw", "tree_hsd", "eagle_tree"]:
    if c in rows:
        r = rows[c]
        print(f"{c:<14} {r['n']:>5} {r['tok_per_step']:>8.3f} {r['tok_per_s']:>7.1f} {r['tok_per_s'] / base if base else float('nan'):>11.2f}x")
