"""Load generator for the oracle study: GSM8K prompts with a K-shot prefix (context length knob),
sent to SGLang's native /generate at a fixed concurrency. Records per-request completion tokens,
verify steps (spec_verify_ct) and latency, plus wall time.

The prompt is sgl-eval's GSM8K prompt (NeMo-Skills generic/math) with K few-shot examples, the
same K examples for every request, so the prefix is shared (RadixAttention caches it) and only the
problem differs. Example solutions are target-only generations from the full GSM8K run.

python oracle_client.py --port 30000 --shots 6 --concurrency 8 --num-requests 24 --out x.json
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SRC = ("runs/gsm8k_bs1/target_only/sgl_eval_gsm8k_20261117-190852/output-rs0.jsonl")
EXAMPLE_POOL_START = 1000  # few-shot examples come from questions 1000+ (correct answers only)
INSTR = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"


def build_prompts(tok, shots, n):
    rows = [json.loads(line) for line in open(SRC)]
    examples = [r for r in rows[EXAMPLE_POOL_START:] if r["symbolic_correct"]][:shots]
    prefix = ""
    if shots:
        prefix = "Here are some examples of problems and solutions you can refer to.\n\n" + "".join(
            f"Problem:\n{r['problem']}\n\nSolution:\n{r['generation']}\n\n\n\n\n\n" for r in examples
        ) + "Here is the problem you need to solve:\n"
    out = []
    for r in rows[:n]:  # benchmark questions: the first n of the test set, same for every config
        text = tok.apply_chat_template(  # the template already has <|begin_of_text|>
            [{"role": "user", "content": INSTR + prefix + r["problem"]}], tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out.append(ids)
    return out


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def verify_counters(port):
    """Lifetime (verified tokens, verify passes) from /server_info; zeros if absent."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/server_info", timeout=60) as resp:
        info = json.load(resp)
    state = info.get("internal_states", [info])[0]
    return state.get("spec_verify_token_ct", 0), state.get("spec_verify_pass_ct", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = build_prompts(tok, args.shots, args.num_requests)
    url = f"http://127.0.0.1:{args.port}/generate"
    sp = {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 1024}
    post(url, {"input_ids": prompts[-1], "sampling_params": {**sp, "max_new_tokens": 8}})  # warm prefix

    def one(ids):
        t0 = time.perf_counter()
        m = post(url, {"input_ids": ids, "sampling_params": sp})["meta_info"]
        return dict(prompt_tokens=len(ids), tokens=m["completion_tokens"],
                    steps=m.get("spec_verify_ct") or m["completion_tokens"],
                    latency=time.perf_counter() - t0)

    tokens0, passes0 = verify_counters(args.port)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        rows = list(pool.map(one, prompts))
    wall = time.perf_counter() - t0
    tokens1, passes1 = verify_counters(args.port)
    tokens = sum(r["tokens"] for r in rows)
    summary = dict(
        shots=args.shots, concurrency=args.concurrency, n=len(rows),
        prompt_tokens=sum(r["prompt_tokens"] for r in rows) / len(rows),
        accept=tokens / sum(r["steps"] for r in rows),
        per_req=tokens / sum(r["latency"] for r in rows), total=tokens / wall, wall=wall,
        # target-verified tokens over the run (0 without the counter / without spec)
        verify_tokens=tokens1 - tokens0, verify_passes=passes1 - passes0,
        verify_tokens_per_pass=(tokens1 - tokens0) / max(passes1 - passes0, 1),
        verify_tokens_per_output=(tokens1 - tokens0) / tokens,
    )
    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
