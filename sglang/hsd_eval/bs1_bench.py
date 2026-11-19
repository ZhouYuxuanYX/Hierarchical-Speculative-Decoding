"""Batch-1 decoding speed in SGLang on the questions of the paper's benchmark (ref_bench.py).

First turn of the first N questions of each task, the same system prompt, T=1, one request at
a time. Reports tokens/s (completion tokens / wall time) and tokens per verify step.
python bs1_bench.py --port 30000 --out runs/bs1/tree_hsd.json
"""

import argparse
import json
import os
import time
import urllib.request

from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATA = (
    "/home/donglab-gpu-local2/project/results_analysis/spd/sglang_spd/"
    "Hierarchical-Speculative-Decoding/EAGLE-3H/eagle/data"
)
TASKS = ["humaneval", "gsm8k", "mt_bench", "sum", "alpaca", "qa"]
# Same default system prompt as the reference benchmark (tree_bench.py).
SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something "
    "not correct. If you don't know the answer to a question, please don't share false information."
)


def _post(url, payload):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--max-questions", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    url = f"http://127.0.0.1:{args.port}/generate"
    prompts = []
    for task in TASKS:
        with open(os.path.join(DATA, task, "question.jsonl")) as f:
            qs = [json.loads(line) for line in f if line.strip()][: args.max_questions]
        for q in qs:
            prompts.append((task, tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["turns"][0]}],
                tokenize=False, add_generation_prompt=True,
            )))
    sp = {"temperature": 1.0, "max_new_tokens": args.max_new_tokens}
    _post(url, {"text": prompts[0][1], "sampling_params": sp})  # warm-up
    rows = []
    for task, text in prompts:
        t0 = time.perf_counter()
        out = _post(url, {"text": text, "sampling_params": sp})
        wall = time.perf_counter() - t0
        m = out["meta_info"]
        rows.append(dict(task=task, tokens=m["completion_tokens"], wall=wall,
                         steps=m.get("spec_verify_ct") or m["completion_tokens"]))
    tokens = sum(r["tokens"] for r in rows)
    summary = dict(
        n=len(rows), tokens=tokens, wall=sum(r["wall"] for r in rows),
        tok_per_s=tokens / sum(r["wall"] for r in rows),
        tok_per_step=tokens / sum(r["steps"] for r in rows),
    )
    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
