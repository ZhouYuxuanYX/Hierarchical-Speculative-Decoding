"""Greedy (T=0) completions of the first N GSM8K prompts at concurrency C (state switching under load).
With --logprobs also saves the target's top-2 log-probs along each output (for the reference run).
python greedy_conc.py --port 30000 --n 64 --concurrency 8 --out runs/accuracy/<cfg>/greedy.json [--logprobs]
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
import oracle_client as oc
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int); ap.add_argument("--n", type=int, default=64)
ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--logprobs", action="store_true")
ap.add_argument("--out", required=True); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(oc.MODEL)
def one(ids):
    p = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 256}}
    if a.logprobs: p.update(return_logprob=True, top_logprobs_num=2)
    r = oc.post(f"http://127.0.0.1:{a.port}/generate", p)
    gaps = None
    if a.logprobs:
        gaps = [(t[0][0] - t[1][0]) if len(t) > 1 else 99.0 for t in r["meta_info"]["output_top_logprobs"]]
    return {"ids": r["output_ids"], "gaps": gaps}
with ThreadPoolExecutor(a.concurrency) as pool:
    outs = list(pool.map(one, oc.build_prompts(tok, 0, a.n)))
json.dump(outs, open(a.out, "w"))
