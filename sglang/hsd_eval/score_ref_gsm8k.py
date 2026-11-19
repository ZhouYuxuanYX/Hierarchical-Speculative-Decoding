"""Score ref_gsm8k.py outputs with sgl-eval's own GSM8K grader and summarize per config.

Run with the sgl-eval venv:
/data/twang/sgl_eval_venv/bin/python score_ref_gsm8k.py runs/ref_gsm8k/shard*.jsonl --out runs/ref_gsm8k/summary.json
--sglang FILE re-scores an sgl-eval output-rs0.jsonl instead (sanity check of the grader path).
"""

import argparse
import asyncio
import collections
import json

import yaml
from sgl_eval._vendored.nemo_skills.evaluator.math import MathEvaluator
from sgl_eval.evals._prompts import resolve_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--sglang")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(resolve_prompt("math")).read()).get("evaluator_config", {})
    ev = MathEvaluator(config=cfg or {})

    def correct(text, answer):
        d = asyncio.run(ev.eval_single({"generation": text, "expected_answer": str(answer)}))
        return bool(d.get("symbolic_correct"))

    if args.sglang:
        rows = [json.loads(line) for line in open(args.sglang)]
        print(f"re-scored {sum(correct(r['generation'], r['expected_answer']) for r in rows) / len(rows):.4%} "
              f"(sgl-eval: {sum(r['symbolic_correct'] for r in rows) / len(rows):.4%}) on {len(rows)}")
        return
    by = collections.defaultdict(list)
    for path in args.files:
        for r in map(json.loads, open(path)):
            by[r["config"]].append(r)
    summary = {}
    for name, rows in by.items():
        tokens, wall, steps = (sum(r[k] for r in rows) for k in ("tokens", "wall", "steps"))
        summary[name] = dict(
            n=len(rows), acc=sum(correct(r["text"], r["expected_answer"]) for r in rows) / len(rows),
            tok_per_step=sum(r["generated"] for r in rows) / steps, tok_per_s=tokens / wall,
            truncated=sum(not r["finished"] for r in rows) / len(rows),
        )
        print(name, json.dumps(summary[name]))
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
