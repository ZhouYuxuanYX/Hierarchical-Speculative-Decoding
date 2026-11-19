"""Greedy (T=0) completions of the first N GSM8K prompts, one at a time; for exactness A/B tests."""
import argparse, json, urllib.request
from transformers import AutoTokenizer
import oracle_client as oc
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int); ap.add_argument("--n", type=int, default=16)
ap.add_argument("--out", required=True); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(oc.MODEL)
outs = []
for ids in oc.build_prompts(tok, 0, a.n):
    r = oc.post(f"http://127.0.0.1:{a.port}/generate",
                {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 256}})
    outs.append(r["output_ids"] if "output_ids" in r else r["text"])
json.dump(outs, open(a.out, "w"))
