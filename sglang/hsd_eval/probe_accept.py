"""Wait for a server, then report tokens per verify step over a few chat prompts.

Run with the sglang_hsd env (needs transformers): python probe_accept.py --port 30000 --temperature 1.0
"""

import argparse
import json
import time
import urllib.request

PROMPTS = [
    "Write a short paragraph about the history of the bicycle.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether? Think step by step.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
]
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _post(url, payload):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def _wait(base, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5):
                return
        except Exception:
            time.sleep(5)
    raise TimeoutError(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--wait", type=int, default=900)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    _wait(base, args.wait)
    from transformers import AutoTokenizer

    # The default system header matters: the EAGLE-3 head was trained with it.
    tok = AutoTokenizer.from_pretrained(MODEL)
    tokens = steps = 0
    for prompt in PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        out = _post(
            base + "/generate",
            {
                "text": text,
                "sampling_params": {
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                },
            },
        )
        tokens += out["meta_info"]["completion_tokens"]
        steps += out["meta_info"]["spec_verify_ct"]
    print(f"port={args.port} T={args.temperature} tokens/step={tokens / steps:.3f}")


if __name__ == "__main__":
    main()
