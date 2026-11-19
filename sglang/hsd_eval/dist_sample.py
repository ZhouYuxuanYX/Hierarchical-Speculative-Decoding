"""Sample the first tokens of many continuations of fixed prompts from a server.

Needs a server launched with --return-output-ids. Saves {prompt_index: [[ids...], ...]}.
python dist_sample.py --port 30000 --out runs/tree_hsd/dist.json
"""

import argparse
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# High-entropy continuations, so draft and target disagree often.
PROMPTS = [
    "Write a list of ten random English words, separated by commas.",
    "Invent a name for a new fantasy city and describe it in one sentence.",
]


def _post(url, payload):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--num-tokens", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--prompts", type=int, nargs="*", help="prompt indices (default: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    url = f"http://127.0.0.1:{args.port}/generate"
    result = {}
    for i, prompt in enumerate(PROMPTS):
        if args.prompts and i not in args.prompts:
            continue
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        samples = []
        while len(samples) < args.num_samples:
            n = min(args.chunk, args.num_samples - len(samples))
            outs = _post(
                url,
                {
                    "text": [text] * n,
                    "sampling_params": {
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "max_new_tokens": args.num_tokens,
                        "ignore_eos": True,
                    },
                },
            )
            samples += [o["output_ids"][: args.num_tokens] for o in outs]
        result[i] = samples
        print(f"prompt {i}: {len(samples)} samples", flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
