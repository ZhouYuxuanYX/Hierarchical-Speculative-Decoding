"""Distribution test under state switching: first-token samples of fixed prompts from single requests
sent by many threads with random lengths, so the running batch (and the adaptive state) keeps changing
while requests are mid-generation. Same prompts / T=1 / top_p=1 as dist_sample.py.
python dist_mixed.py --port 30000 --out runs/accuracy/<cfg>/dist.json
"""
import argparse, json, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from dist_sample import MODEL, PROMPTS
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int); ap.add_argument("--num-samples", type=int, default=3000)
ap.add_argument("--num-tokens", type=int, default=8); ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--prompts", type=int, nargs="*", help="prompt indices (default: all)")
ap.add_argument("--out", required=True); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(MODEL)
url = f"http://127.0.0.1:{a.port}/generate"
def one(text, seed):
    rng = random.Random(seed)
    sp = {"temperature": a.temperature, "top_p": 1.0, "ignore_eos": True,
          "max_new_tokens": a.num_tokens + rng.randint(0, 40)}  # varied lifetimes -> batch churn
    req = urllib.request.Request(url, json.dumps({"text": text, "sampling_params": sp}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))["output_ids"][: a.num_tokens]
result = {}
for i, prompt in enumerate(PROMPTS):
    if a.prompts and i not in a.prompts:
        continue
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    with ThreadPoolExecutor(a.threads) as pool:
        result[i] = list(pool.map(lambda s: one(text, s), range(i * 10**6, i * 10**6 + a.num_samples)))
    print(f"prompt {i}: {len(result[i])} samples", flush=True)
json.dump(result, open(a.out, "w"))
