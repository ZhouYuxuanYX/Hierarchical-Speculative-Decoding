"""Check (1): same 5 inputs, near-deterministic sampling -> spec outputs must equal target greedy.

T=0 runs SGLang's greedy verify (the HSD kernel is not called); T=0.01 runs the sampled-tree /
HSD path while the target is almost one-hot. Requests go one at a time (batch size 1).
python check_greedy.py --port 30001 --out runs/greedy/tree_hsd.json [--logprobs]
"""

import argparse
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
    "in May. How many clips did Natalia sell altogether in April and May? Think step by step.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain why the sky is blue in three sentences.",
    "Write a list of ten random English words, separated by commas.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
]


def _post(url, payload):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--logprobs", action="store_true", help="also save target top-2 logprobs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    url = f"http://127.0.0.1:{args.port}/generate"
    result = {}
    for temperature in (0.0, 0.01):
        rows = []
        for prompt in PROMPTS:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            payload = {
                "text": text,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            if args.logprobs and temperature == 0.0:
                payload.update(return_logprob=True, top_logprobs_num=2)
            out = _post(url, payload)
            row = {"ids": out["output_ids"], "meta": {
                k: out["meta_info"].get(k) for k in ("spec_accept_length", "spec_verify_ct")
            }}
            if "output_top_logprobs" in out["meta_info"]:
                row["top2"] = [[lp[0] for lp in pos] for pos in out["meta_info"]["output_top_logprobs"]]
            rows.append(row)
        result[str(temperature)] = rows
    with open(args.out, "w") as f:
        json.dump(result, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
