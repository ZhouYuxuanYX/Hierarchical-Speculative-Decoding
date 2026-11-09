"""Sampled vs top-k EAGLE-3 draft trees with target and draft at temperature 1, on one GPU.

Loads Llama-3.1-8B-Instruct + the EAGLE3 head once. For every question of this shard it runs each
config and seed back to back (config order rotates per question), so all configs see the same GPU:
  eagle         EAGLE's own tree: top-10 children, depth 5, pruned to 59 nodes; token-wise verification
  topk          top-4 children, depth 5, all 84 nodes kept (EAGLE's builder); token-wise verification
  sampled_rrsw  the same shape with children drawn from q without replacement; rrsw verification
  sampled_hsd   the same shape and draws; tree HSD verification
  spine_rrsw    sampled, but each level expands the first-drawn child of every node expanded above
  spine_hsd     (so every root child heads a 6-token path of first children); rrsw / tree HSD
  hybrid1_*     sampled; per level 1 of the 4 expansions continues the path of first children under
  hybrid2_*     the first root child (2: under the first two), the rest by path log q; rrsw / HSD
  hybrid1_hsdf  hybrid1 with the fused HSD walker (one Triton kernel per path), never shadowed: timing
  deep2_*       hybrid1 whose first spine goes on for 2 more levels (8 drafted tokens on that path, 92 nodes)
  deep4_*       the same with 4 more levels (10 drafted tokens, 100 nodes); rrsw / fused HSD (the *_hsdf
                configs are shadowed with --shadow, fused on the HSD side)
All other trees have draft_length 7 (the root token plus up to 6 drafted tokens per path), as in EAGLE-3H.
Token-wise is EAGLE's evaluate_posterior (q = 1 on the drafted token), lossless for deterministic
children; the sampled configs use sampled_tree.py. With --shadow N the *_hsd configs also run both
walkers N more times per step on the same tree with separate coins (paired comparison; the generation
is unchanged). Writes one JSON line per generated turn; re-running skips lines already written.
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

p = argparse.ArgumentParser()
p.add_argument("--code", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "EAGLE-3H"),
               help="EAGLE-3H directory (default: the one in this repository)")
p.add_argument("--base", required=True)
p.add_argument("--ea", required=True)
p.add_argument("--tasks", default="humaneval,gsm8k,mt_bench,sum,alpaca,qa")
p.add_argument("--configs", default="eagle,topk,sampled_rrsw,sampled_hsd")
p.add_argument("--shadow", type=int, default=0)
p.add_argument("--seeds", default="0,1")
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--max-questions", type=int, default=80)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num-shards", type=int, default=1)
p.add_argument("--out", required=True)
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

sys.path.insert(0, args.code)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eagle.model.ea_model as ea_mod  # noqa: E402
import eagle.model.utils as eagle_utils  # noqa: E402
import sampled_tree as st  # noqa: E402
from eagle.model.cnets import Model  # noqa: E402

spine = functools.partial(st.topK_genrate, spines=4)
hybrid1 = functools.partial(st.topK_genrate, spines=1)
hybrid2 = functools.partial(st.topK_genrate, spines=2)
deep2 = functools.partial(st.topK_genrate, spines=1, extra=2)
deep4 = functools.partial(st.topK_genrate, spines=1, extra=4)
# name: (children per node, depth, nodes kept, tree builder, verifier)
CONFIGS = {
    "eagle": (10, 5, 59, Model.topK_genrate, eagle_utils.evaluate_posterior),
    "topk": (4, 5, 84, Model.topK_genrate, eagle_utils.evaluate_posterior),
    "sampled_rrsw": (4, 5, 84, st.topK_genrate, st.make_evaluate_posterior(st.rrsw)),
    "sampled_hsd": (4, 5, 84, st.topK_genrate, st.make_evaluate_posterior(st.hsd, args.shadow)),
    "spine_rrsw": (4, 5, 84, spine, st.make_evaluate_posterior(st.rrsw)),
    "spine_hsd": (4, 5, 84, spine, st.make_evaluate_posterior(st.hsd, args.shadow)),
    "hybrid1_rrsw": (4, 5, 84, hybrid1, st.make_evaluate_posterior(st.rrsw)),
    "hybrid1_hsd": (4, 5, 84, hybrid1, st.make_evaluate_posterior(st.hsd, args.shadow)),
    "hybrid2_rrsw": (4, 5, 84, hybrid2, st.make_evaluate_posterior(st.rrsw)),
    "hybrid2_hsd": (4, 5, 84, hybrid2, st.make_evaluate_posterior(st.hsd, args.shadow)),
    "hybrid1_hsdf": (4, 5, 84, hybrid1, st.make_evaluate_posterior(st.hsd_fused)),
    "deep2_rrsw": (4, 5, 92, deep2, st.make_evaluate_posterior(st.rrsw)),
    "deep2_hsdf": (4, 5, 92, deep2, st.make_evaluate_posterior(st.hsd_fused, args.shadow, shadow_hsd=st.hsd_fused)),
    "deep4_rrsw": (4, 5, 100, deep4, st.make_evaluate_posterior(st.rrsw)),
    "deep4_hsdf": (4, 5, 100, deep4, st.make_evaluate_posterior(st.hsd_fused, args.shadow, shadow_hsd=st.hsd_fused)),
}
NAMES = args.configs.split(",")
# Same default system prompt as eagle/evaluation/gen_ea_answer_llama3chat.py
SYSTEM = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
          "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
          "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
          "If a question does not make any sense, or is not factually coherent, explain why instead of answering something "
          "not correct. If you don't know the answer to a question, please don't share false information.")

data = os.path.join(args.code, "eagle", "data")
items = []
for task in args.tasks.split(","):
    with open(os.path.join(data, task, "question.jsonl")) as f:
        qs = [json.loads(line) for line in f if line.strip()][: args.max_questions]
    items += [(task, q) for q in qs]
items = items[args.shard :: args.num_shards]
seeds = [int(s) for s in args.seeds.split(",")]
done = set()
if os.path.exists(args.out):
    with open(args.out) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["task"], r["question_id"], r["seed"], r["config"]))
print(f"shard {args.shard}/{args.num_shards}: {len(items)} questions x {len(seeds)} seeds x {len(NAMES)} configs, "
      f"{len(done)} already done", flush=True)
if args.dry_run:
    sys.exit(0)

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


def generate(turns, name, seed):
    configure(name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    messages = [{"role": "system", "content": SYSTEM}]
    recs = []
    for qs in turns:
        messages.append({"role": "user", "content": qs})
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok([prompt], add_special_tokens=False).input_ids
        st.STATS.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, new_token, _, info = model.eagenerate(torch.as_tensor(ids).cuda(), temperature=args.temperature,
                                                   log=True, is_llama3=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        gen = out[0][len(ids[0]):].tolist()
        cut = next((i for i, t in enumerate(gen) if t in stops), len(gen))
        text = tok.decode(gen[:cut], skip_special_tokens=True).strip()
        acc = [int(a) for a in info["accept_length"]]
        rec = dict(new_tokens=int(new_token), steps=len(acc), wall=wall, accept=acc,
                   target_fwd=float(sum(info["tree_time"])), verify=float(sum(info["eval_time"])),
                   update=float(sum(info["update_time"])), text=text)
        if st.STATS:  # sampled configs: spine length per step; shadow sums (their time is in wall and verify)
            rec.update(spine=[s["spine"] for s in st.STATS], shadow_time=sum(s["shadow_s"] for s in st.STATS))
            if "hsd" in st.STATS[0]:
                rec.update(shadow_rrsw=[s["rrsw"] for s in st.STATS], shadow_hsd=[s["hsd"] for s in st.STATS])
        recs.append(rec)
        messages.append({"role": "assistant", "content": text})
    return recs


for name in NAMES:  # warm-up
    generate(items[0][1]["turns"][:1], name, 0)
print("warm-up done", flush=True)

with open(args.out, "a") as f:
    for i, (task, q) in enumerate(items):
        for seed in seeds:
            k = i % len(NAMES)
            for name in NAMES[k:] + NAMES[:k]:
                if (task, q["question_id"], seed, name) in done:
                    continue
                for turn, r in enumerate(generate(q["turns"], name, seed)):
                    f.write(json.dumps(dict(task=task, question_id=q["question_id"], turn=turn, config=name,
                                            seed=seed, shard=args.shard, **r)) + "\n")
                f.flush()
        print(f"{i + 1}/{len(items)} {task} q{q['question_id']} done", flush=True)
print("shard finished", flush=True)
