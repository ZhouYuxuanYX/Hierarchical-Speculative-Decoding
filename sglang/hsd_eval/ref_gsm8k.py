"""GSM8K with the paper's reference code (no SGLang), batch 1, matched to the SGLang runs.

Same 1319 questions, same user prompt as sgl-eval (NeMo-Skills generic/math, no system message),
T=0.6, top-p 0.9, up to 1024 new tokens. Each question runs every config back to back (order rotates
per question), so all configs see the same GPU. Score with score_ref_gsm8k.py (sgl-eval's grader).

Config <-> SGLang run:
  target       <-> target_only    (naivegenerate: plain autoregressive decoding)
  chain_rrsw   <-> chain_token    (1 child per node, 6 drafted tokens, token-wise RS)
  chain_hsd    <-> chain_hsd      (same chain, HSD = block verification)
  sampled_rrsw <-> tree_rrsw      (k=4, depth 5, 84 drafted nodes, sampled without replacement)
  sampled_hsd  <-> tree_hsd       (same tree, tree HSD)
  eagle        <-> eagle_tree     (stock EAGLE-3 top-k tree, k=10, depth 5, 59 nodes)

python ref_gsm8k.py --base ... --ea ... --shard 0 --num-shards 2 --out runs/ref_gsm8k/shard0.jsonl
"""

import argparse
import functools
import json
import os
import random
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE_HSD = os.path.join(HERE, "..", "Hierarchical-Speculative-Decoding", "tree-hsd")
QUESTIONS = os.path.join(  # problems + expected answers in sgl-eval's order
    HERE, "runs/gsm8k_bs1/target_only/sgl_eval_gsm8k_20261117-190852/output-rs0.jsonl"
)
PROMPT = (  # sgl-eval gsm8k user prompt (NeMo-Skills prompts/math.yaml, no few-shot examples)
    "Solve the following math problem. Make sure to put the answer (and only answer) inside "
    "\\boxed{{}}.\n\n{problem}"
)

p = argparse.ArgumentParser()
p.add_argument("--code", default=os.path.join(TREE_HSD, "..", "EAGLE-3H"))
p.add_argument("--base", required=True)
p.add_argument("--ea", required=True)
p.add_argument("--configs", default="target,chain_rrsw,chain_hsd,sampled_rrsw,sampled_hsd,eagle")
p.add_argument("--temperature", type=float, default=0.6)
p.add_argument("--top-p", type=float, default=0.9)
p.add_argument("--max-new-tokens", type=int, default=1024)
p.add_argument("--max-questions", type=int, default=None)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num-shards", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
args = p.parse_args()

sys.path.insert(0, args.code)
sys.path.insert(0, TREE_HSD)
import eagle.model.ea_model as ea_mod  # noqa: E402
import eagle.model.utils as eagle_utils  # noqa: E402
import sampled_tree as st  # noqa: E402
from eagle.model.cnets import Model  # noqa: E402

# name: (children per node, depth, nodes kept, tree builder, verifier); as in ref_bench.py
CONFIGS = {
    "eagle": (10, 5, 59, Model.topK_genrate, eagle_utils.evaluate_posterior),
    "chain_rrsw": (1, 5, 6, st.topK_genrate, st.make_evaluate_posterior(st.rrsw)),
    "chain_hsd": (1, 5, 6, st.topK_genrate, st.make_evaluate_posterior(st.hsd)),
    "sampled_rrsw": (4, 5, 84, st.topK_genrate, st.make_evaluate_posterior(st.rrsw)),
    "sampled_hsd": (4, 5, 84, st.topK_genrate, st.make_evaluate_posterior(st.hsd)),
}
NAMES = args.configs.split(",")
assert all(n == "target" or n in CONFIGS for n in NAMES), NAMES

with open(QUESTIONS) as f:
    items = [json.loads(line) for line in f if line.strip()][: args.max_questions]
items = [(i, r["problem"], r["expected_answer"]) for i, r in enumerate(items)]
items = items[args.shard :: args.num_shards]
done = set()
if os.path.exists(args.out):
    with open(args.out) as f:
        done = {(r["idx"], r["config"]) for r in map(json.loads, f)}
print(f"shard {args.shard}/{args.num_shards}: {len(items)} questions x {len(NAMES)} configs, "
      f"{len(done)} already done", flush=True)

model = ea_mod.EaModel.from_pretrained(base_model_path=args.base, ea_model_path=args.ea, total_token=60, depth=5,
                                       top_k=10, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                       device_map="auto", use_eagle3=True)
model.eval()
tok = model.get_tokenizer()
stops = {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")}


def configure(name):
    k, depth, nodes, builder, verify = CONFIGS[name]
    layer = model.ea_layer
    layer.top_k, layer.depth, layer.total_tokens = k, depth, nodes
    layer.init_tree()
    layer.topK_genrate = types.MethodType(builder, layer)
    ea_mod.evaluate_posterior = verify


def generate(problem, name):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=problem)}],
                                     tokenize=False, add_generation_prompt=True)
    ids = torch.as_tensor(tok([prompt], add_special_tokens=False).input_ids).cuda()
    kw = dict(temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
              max_length=2048, log=True, is_llama3=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if name == "target":
        out, _, steps, _ = model.naivegenerate(ids, **kw)
        steps = int(steps) + 1
    else:
        configure(name)
        out, _, _, info = model.eagenerate(ids, **kw)
        steps = len(info["accept_length"])
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gen = out[0][ids.shape[1]:].tolist()
    cut = next((i for i, t in enumerate(gen) if t in stops), len(gen))
    gen = gen[: min(cut, args.max_new_tokens)]
    return dict(tokens=len(gen), generated=len(out[0]) - ids.shape[1], steps=steps, wall=wall,
                finished=cut < len(out[0]) - ids.shape[1], text=tok.decode(gen, skip_special_tokens=True))


for name in NAMES:  # warm-up
    generate(items[0][1], name)
print("warm-up done", flush=True)

with open(args.out, "a") as f:
    for n, (idx, problem, answer) in enumerate(items):
        k = n % len(NAMES)
        for name in NAMES[k:] + NAMES[:k]:
            if (idx, name) in done:
                continue
            r = generate(problem, name)
            f.write(json.dumps(dict(idx=idx, config=name, expected_answer=answer, **r)) + "\n")
            f.flush()
        print(f"{n + 1}/{len(items)} q{idx} done", flush=True)
print("shard finished", flush=True)
