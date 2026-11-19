"""Check (2): exact losslessness of naive tree HSD on real Llama-3.1-8B / EAGLE-3 distributions,
and our Triton kernel against the naive per-tree law.

Model: at a context c (prompt + real tokens), p(.|c) is the target's distribution and q(.|c) the
EAGLE-3 head's proposal for the token after c, with the last token of c as the target's emitted
token (as in EAGLE's own drafting; dumped by the debug server). Both are restricted to
the alphabet A(c) = top-m tokens by (p + q) / 2 and renormalized; symbol i at c means token A(c)[i].
On this model the paper's exact checker (tree-hsd/verify_sampled_tree_wor.py) enumerates every
sampled tree and every walk outcome:
  (a) TV(output law of hsd_tree_wor / rrsw_tree, target joint)   -> should be ~1e-15
  (b) contrast: the same verifier on top-k (unsampled) trees       -> should be clearly > 0
  (c) kernel: sampled_tree_verify on the most likely trees, many coins, vs the exact law of
      hsd_tree_wor / rrsw_tree on the same tree (TV, chi-square p)

Needs a server from the debug worktree with SGLANG_DEBUG_DUMP_ROOT_Q set (see run_exact.sh).
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import uuid

import numpy as np
import torch
from scipy.stats import chisquare

sys.path.insert(
    0,
    "/home/donglab-gpu-local2/project/results_analysis/spd/sglang_spd/"
    "Hierarchical-Speculative-Decoding/tree-hsd",
)
from verify_hsd_exact import target_joint, tv  # noqa: E402
from verify_sampled_tree_wor import (  # noqa: E402
    horizon,
    hsd_tree_wor,
    mean_len,
    rrsw_tree,
    step_law,
    trees,
)

from sglang.kernels.ops.speculative.sampled_tree_verify import (  # noqa: E402
    sampled_tree_verify,
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS = [  # the same 5 inputs as check (1)
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
    "in May. How many clips did Natalia sell altogether in April and May? Think step by step.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain why the sky is blue in three sentences.",
    "Write a list of ten random English words, separated by commas.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
]
Q_FLOOR = 1e-6  # q outside the dumped top-256: tiny, so every symbol stays a possible draw


class RealEnv:
    """Env interface of the paper's checker (.V, .pq(ctx)) backed by the real models."""

    def __init__(self, port, prompt_ids, m, dump_dir):
        self.V, self.port, self.prompt_ids, self.dump_dir = m, port, prompt_ids, dump_dir
        self.real = {}  # real context tuple -> (p over A, q over A, A)
        self.requests = 0  # HTTP requests, two per context

    def _post(self, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/generate",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        self.requests += 1
        return json.load(urllib.request.urlopen(req, timeout=600))

    def _fetch(self, real_ctx):
        full = self.prompt_ids + list(real_ctx)
        # p(.|c): the target's next-token log-probs after prefilling c.
        out = self._post({
            "input_ids": full,
            "sampling_params": {"temperature": 1.0, "max_new_tokens": 1},
            "return_logprob": True,
            "top_logprobs_num": 256,
        })
        p = {tok: math.exp(lp) for lp, tok, *_ in out["meta_info"]["output_top_logprobs"][0]}
        # q(.|c): after a prefill EAGLE drafts the token after the target's sampled token, so
        # prefill c[:-1] and force the target to emit c[-1]; the dump then holds q(.|c).
        rid = uuid.uuid4().hex
        out = self._post({
            "input_ids": full[:-1],
            "rid": rid,
            "sampling_params": {
                "temperature": 1.0,
                "max_new_tokens": 1,
                "logit_bias": {str(full[-1]): 1e4},
            },
        })
        assert out["output_ids"][0] == full[-1], (out["output_ids"], full[-1])
        path = f"{self.dump_dir}/{rid}.json"
        for _ in range(200):
            if os.path.exists(path):
                break
            time.sleep(0.05)
        d = json.load(open(path))
        return p, dict(zip(d["ids"], d["q"]))

    def _entry(self, real_ctx):
        if real_ctx not in self.real:
            p, q = self._fetch(real_ctx)
            score = {t: p.get(t, 0.0) + q.get(t, 0.0) for t in set(p) | set(q)}
            alphabet = sorted(score, key=lambda t: -score[t])[: self.V]
            pv = np.array([p.get(t, 0.0) for t in alphabet])
            qv = np.array([max(q.get(t, 0.0), Q_FLOOR) for t in alphabet])
            self.real[real_ctx] = (pv / pv.sum(), qv / qv.sum(), alphabet)
        return self.real[real_ctx]

    def real_tokens(self, ctx):
        real = ()
        for sym in ctx:
            real = real + (self._entry(real)[2][sym],)
        return real

    def pq(self, ctx):
        p, q, _ = self._entry(self.real_tokens(ctx))
        return p, q


def kernel_inputs(env, tree, W, depth, reps):
    """EAGLE layout of a tree: group g's children are nodes 1 + g*W .. 1 + g*W + W - 1."""
    n_nodes = 1 + W + (depth - 1) * W * W
    prefix_of, tokens = {0: ()}, [0] * n_nodes
    first, sibling = [-1] * n_nodes, [-1] * n_nodes
    groups, node_id = [()], {(): 0}
    for level in range(depth):
        parents = [()] if level == 0 else sorted(
            (p for p in tree if len(p) == level), key=lambda p: node_id[p]
        )
        for parent in parents:
            g = 0 if level == 0 else len(groups)
            if level > 0:
                groups.append(parent)
            kids = tree[parent]
            for c, sym in enumerate(kids):
                n = 1 + g * W + c
                node_id[parent + (sym,)] = n
                prefix_of[n], tokens[n] = parent + (sym,), sym
                if c + 1 < len(kids):
                    sibling[n] = n + 1
            first[node_id[parent]] = 1 + g * W
    target = np.stack([env.pq(prefix_of.get(n, ()))[0] for n in range(n_nodes)])
    draft = np.stack([env.pq(p)[1] for p in groups])
    dev = "cuda"

    def rep(a, dtype):
        t = torch.tensor(np.asarray(a), dtype=dtype, device=dev)
        return t.unsqueeze(0).expand(reps, *t.shape).contiguous()

    offsets = (torch.arange(reps, device=dev) * n_nodes)[:, None]
    return dict(
        candidates=rep(tokens, torch.int64),
        retrieve_index=offsets + torch.arange(n_nodes, device=dev),
        next_token=rep(first, torch.int64),
        next_sibling=rep(sibling, torch.int64),
        target_probs=rep(target, torch.float32),
        draft_probs=rep(draft, torch.float32),
        n_nodes=n_nodes,
    )


def kernel_law_check(env, tree, W, depth, use_hsd, reps):
    k = kernel_inputs(env, tree, W, depth, reps)
    n = k["n_nodes"]
    predicts = torch.full((reps * n,), -1, dtype=torch.int32, device="cuda")
    accept = torch.full((reps, depth + 1), -1, dtype=torch.int32, device="cuda")
    num = torch.empty(reps, dtype=torch.int32, device="cuda")
    sampled_tree_verify(
        predicts, accept, num, k["candidates"], k["retrieve_index"], k["next_token"],
        k["next_sibling"], torch.rand(reps, n, depth + 1, device="cuda"),
        torch.rand(reps, device="cuda"), k["target_probs"], k["draft_probs"],
        topk=W, use_hsd=use_hsd,
    )
    local = (accept.long() - (torch.arange(reps, device="cuda") * n)[:, None]).clamp(min=0)
    toks = k["candidates"].gather(1, local)
    bonus = predicts.long()[accept.long().gather(1, num.long()[:, None])][:, 0]
    seqs = torch.full((reps, depth + 1), -1, dtype=torch.long, device="cuda")
    pos = torch.arange(depth + 1, device="cuda")[None]
    drafted = torch.cat([toks[:, 1:], toks[:, :1]], 1)  # column j: (j+1)-th accepted node
    seqs = torch.where(pos < num[:, None], drafted, seqs)
    seqs = torch.where(pos == num[:, None], bonus[:, None], seqs)
    rows, counts = np.unique(seqs.cpu().numpy(), axis=0, return_counts=True)
    empirical = {tuple(int(x) for x in r if x >= 0): c / reps for r, c in zip(rows, counts)}
    exact = (hsd_tree_wor if use_hsd else rrsw_tree)(env, tree)
    keys = [s for s in exact if exact[s] * reps >= 5]
    obs = [empirical.get(s, 0.0) * reps for s in keys]
    exp = [exact[s] * reps for s in keys]
    if reps - sum(exp) >= 5:  # pool the rare outcomes into one category
        obs.append(reps - sum(obs))
        exp.append(reps - sum(exp))
    exp = [e * sum(obs) / sum(exp) for e in exp]
    pval = chisquare(obs, exp).pvalue if len(keys) > 1 else 1.0
    return tv(empirical, exact), pval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--reps", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    # (m, depth, W, E): alphabet size, draft depth, children per node, expansions per level
    shapes = [(4, 2, 2, 2), (3, 3, 2, 2)]
    report = []
    for pi, prompt in enumerate(PROMPTS):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]  # the template has <|begin_of_text|>
        for m, depth, W, E in shapes:
            env = RealEnv(args.port, ids, m, args.dump_dir)
            L = depth + 1
            target = target_joint(env, (), L)
            row = dict(prompt=pi, m=m, depth=depth, W=W, E=E)
            for name, f in (("hsd", hsd_tree_wor), ("rrsw", rrsw_tree)):
                row[f"tv_{name}"] = tv(horizon(env, (), L, depth, W, E, f, True, {}), target)
                row[f"len_{name}"] = mean_len(step_law(env, (), depth, W, E, f, True))
            row["tv_hsd_on_topk_trees"] = tv(
                horizon(env, (), L, depth, W, E, hsd_tree_wor, False, {}), target
            )
            likely = sorted(trees(env, depth, W, E), key=lambda t: -t[1])[:3]
            row["kernel"] = []
            for tree, prob in likely:
                for use_hsd in (True, False):
                    d, pval = kernel_law_check(env, tree, W, depth, use_hsd, args.reps)
                    row["kernel"].append(
                        dict(tree_prob=prob, hsd=use_hsd, tv=d, chi2_p=pval)
                    )
            row["contexts"] = len(env.real)
            report.append(row)
            worst_k = min(r["chi2_p"] for r in row["kernel"])
            print(
                f"prompt {pi} m={m} depth={depth}: TV hsd={row['tv_hsd']:.1e} "
                f"rrsw={row['tv_rrsw']:.1e} | hsd on top-k trees={row['tv_hsd_on_topk_trees']:.1e} "
                f"| tokens/step hsd={row['len_hsd']:.4f} rrsw={row['len_rrsw']:.4f} "
                f"| kernel vs naive: min chi2 p={worst_k:.3f} max TV="
                f"{max(r['tv'] for r in row['kernel']):.1e} | {len(env.real)} contexts",
                flush=True,
            )
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
