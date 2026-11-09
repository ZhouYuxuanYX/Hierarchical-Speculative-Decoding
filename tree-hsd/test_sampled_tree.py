"""CPU checks for sampled_tree.py.

A. The rrsw and hsd walkers against the exact one-step laws of _hsd/verify_sampled_tree_wor.py:
   Monte Carlo over sampled trees on small random (p, q) environments. Reports the TV distance to
   the exact law next to the TV of the same number of draws from the exact law itself, the largest
   per-outcome z-score, and tokens/step. A1 / A2 do the same with 1 / 2 expansions per level kept on
   paths of first-drawn children (spines=1, 2); AX with spines=1 and 2 extra levels on the first path.
B. With the real EAGLE3 head on CPU (float32): topK_genrate(sample=False) reproduces EAGLE's own
   builder run without pruning; the sampled trees (spines = 0, 1, 2, k, and spines 1 / 2 with extra
   levels) have the intended structure; the evaluate_posterior adapter returns consistent rows and
   records the per-step stats.
C. children(sample=True) draws ordered children with the without-replacement law.
F. hsd_fused (Triton, run by the interpreter on CPU) walks exactly like hsd with the same coins, on
   the environments of A (with and without extra levels) and on real-head trees.
Usage: test_sampled_tree.py [A|A1|A2|AX|B|C|F ...] [--mc N]
"""
import collections
import os
import random
import sys
import time

import numpy as np
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")  # before sampled_tree imports Triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # the exact checks
import sampled_tree as st  # noqa: E402
from verify_hsd_exact import Env  # noqa: E402
from verify_sampled_tree_wor import (hsd_tree_wor, path_logq, rrsw_tree, spine_nodes, step_law,  # noqa: E402
                                     wor_sequences)

CODE = os.path.join(HERE, "..", "EAGLE-3H")
# part B: local copies of lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B and meta-llama/Llama-3.1-8B-Instruct
EA = os.environ.get("EAGLE3_HEAD", "")
BASE = os.environ.get("BASE_MODEL", "")


def sample_tree(env, depth, W, rng, spines=0, extra=0):
    """One draw of the tree process verify_sampled_tree_wor.trees(env, depth, W, W, spines, extra) enumerates."""

    def draw(q):
        return tuple(np.argsort(-(np.log(q) + rng.gumbel(size=len(q))), kind="stable")[:W].tolist())

    tree = {(): draw(env.pq(())[1])}
    for level in range(1, depth + extra):
        if level >= depth:  # extra levels: only the first path of first-drawn children goes on
            n = spine_nodes(tree, level, 1)[0]
            tree[n] = draw(env.pq(n)[1])
            continue
        cand = [p + (c,) for p, kids in tree.items() for c in kids if len(p) + 1 == level]
        spine = spine_nodes(tree, level, spines)
        rest = sorted([p for p in cand if p not in spine], key=lambda p: -path_logq(env, p))
        for n in spine + rest[: W - len(spine)]:
            tree[n] = draw(env.pq(n)[1])
    return tree


class EnvView:
    """A tree over prefixes of env, with the tables the walkers read (see sampled_tree.TreeView)."""

    def __init__(self, env, tree):
        prefix, kids, tokens = [()], [[]], [0]
        queue = [0]
        while queue:  # breadth first, so children get larger ids than their parent
            u = queue.pop(0)
            for c in tree.get(prefix[u], ()):
                kids[u].append(len(prefix))
                queue.append(len(prefix))
                prefix.append(prefix[u] + (c,))
                kids.append([])
                tokens.append(c)
        expanded = [u for u in range(len(prefix)) if kids[u]]
        qrow = [-1] * len(prefix)
        for r, u in enumerate(expanded):
            qrow[u] = r
        self.prefix = prefix
        self.__dict__.update(st.tables(kids, tokens, qrow))
        self.P = torch.tensor(np.array([env.pq(p)[0] for p in prefix]))
        self.Q = torch.tensor(np.array([env.pq(prefix[u])[1] for u in expanded]))

    def p(self, u):
        return self.P[u]

    def q(self, u):
        return self.Q[self.qrow[u]]


def check_walkers(M, spines=0, extra=0):
    cases = [(3, 3, 2, 0.5, 0.0, 0), (4, 2, 2, 1.0, 0.6, 1), (5, 2, 2, 0.3, 0.5, 2), (4, 3, 2, 0.5, 0.3, 3)]
    if extra:  # the extra levels start right below depth
        cases = [(3, 2, 2, 0.5, 0.0, 0), (4, 2, 2, 1.0, 0.6, 1)]
    elif spines:  # the expansion rules differ only from the third level on
        cases = [(3, 3, 2, 0.5, 0.0, 0), (4, 3, 2, 0.5, 0.3, 3)]
    for V, depth, W, alpha, mix, seed in cases:  # depth = levels of draft tokens; W children per node
        env = Env(V, seed, alpha, alpha, 0.0, mix)
        for name, walk, exact_fn in (("rrsw", st.rrsw, rrsw_tree), ("hsd", st.hsd, hsd_tree_wor)):
            exact = step_law(env, (), depth, W, W, exact_fn, spines=spines, extra=extra)
            rng, py = np.random.default_rng(seed), random.Random(seed)
            counts, t0 = collections.Counter(), time.time()
            for _ in range(M):
                view = EnvView(env, sample_tree(env, depth, W, rng, spines, extra))
                path, R = walk(view, py.random)
                R = R.numpy()
                y = int(rng.choice(V, p=R / R.sum()))
                counts[tuple(view.tokens[v] for v in path) + (y,)] += 1
            keys = sorted(set(exact) | set(counts))
            e = np.array([exact.get(k, 0.0) for k in keys])
            f = np.array([counts.get(k, 0) for k in keys]) / M
            ref = rng.multinomial(M, e / e.sum()) / M  # M draws from the exact law: the noise floor
            big = M * e >= 5  # chi-square over outcomes expected at least 5 times, the rest pooled
            obs = np.append(M * f[big], M * f[~big].sum())
            exp = np.append(M * e[big], M * e[~big].sum())
            keep = exp > 0
            chi2 = (((obs - exp) ** 2)[keep] / exp[keep]).sum()
            dof = keep.sum() - 1
            z = np.abs(f - e)[big] / np.sqrt(e[big] * (1 - e[big]) / M)
            lens = np.array([len(k) for k in keys])
            print(f"V={V} depth={depth} W={W} spines={spines} extra={extra} alpha={alpha} mix={mix} {name:<4}: TV "
                  f"{0.5 * np.abs(f - e).sum():.4f} "
                  f"(exact-law draws {0.5 * np.abs(ref - e).sum():.4f}) chi2/dof {chi2 / dof:.2f} (dof {dof}, "
                  f"sd {np.sqrt(2 / dof):.2f}) max|z| {z.max():.2f}, {int(f[e == 0].sum() * M)} impossible | "
                  f"tokens/step {f @ lens:.4f} vs exact {e @ lens:.4f} | {time.time() - t0:.0f}s", flush=True)


def check_builder():
    if not (EA and BASE):
        print("part B skipped: set EAGLE3_HEAD and BASE_MODEL to the local model directories")
        return
    sys.path.insert(0, CODE)
    from transformers import LogitsProcessorList
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig

    k, depth = 4, 5
    n = k + depth * k * k
    config = EConfig.from_pretrained(EA)
    layer = Model(config, total_tokens=n + 1, depth=depth, top_k=k, path=BASE, load_emb=True)
    print("load_state_dict:", layer.load_state_dict(torch.load(os.path.join(EA, "pytorch_model.bin"),
                                                               map_location="cpu"), strict=False))
    layer.float().eval()
    layer.init_tree()
    g = torch.Generator().manual_seed(0)
    L = 12
    hidden = torch.randn(1, L, 3 * config.hidden_size, generator=g)
    ids = torch.randint(0, 128000, (1, L + 1), generator=g)
    hidden2 = torch.randn(1, 3, 3 * config.hidden_size, generator=g)  # second call goes through stable_kv
    ids2 = torch.cat([ids, torch.randint(0, 128000, (1, 3), generator=g)], dim=1)
    names = ("draft_tokens", "retrieve_indices", "tree_mask", "tree_position_ids")
    for lp in (LogitsProcessorList(), None):
        outs = []
        for fn in (lambda h, x: Model.topK_genrate(layer, h, x, None, lp),
                   lambda h, x: st.topK_genrate(layer, h, x, None, lp, sample=False)):
            layer.reset_kv()
            outs.append([fn(hidden, ids), fn(hidden2, ids2)])
        for call in range(2):
            for name, a, b in zip(names, outs[0][call], outs[1][call]):
                assert a.shape == b.shape and torch.equal(a, b), (call, name, a.shape, b.shape)
        print(f"top-k mode == EAGLE's builder with total_tokens={n} ({'sampling' if lp is not None else 'greedy'} "
              f"row order, both calls): ok; {outs[1][0][1].shape[0]} leaves, width {outs[1][0][1].shape[1]}")

    lp = LogitsProcessorList()
    for spines, extra in ((0, 0), (1, 0), (2, 0), (k, 0), (1, 2), (2, 3)):
        layer.reset_kv()
        torch.manual_seed(1)
        draft, rows_t, mask_t, pos_t = st.topK_genrate(layer, hidden, ids, None, lp, sample=True, spines=spines,
                                                       extra=extra)
        n = k + depth * k * k + extra * k
        S = st.SIDE
        toks, kids, pos, rows = S["tokens"], S["kids"], pos_t.tolist(), rows_t.tolist()
        assert len(toks) == n + 1 and toks == draft[0].tolist() and toks[0] == ids[0, -1].item()
        parent = {v: u for u, ks in enumerate(kids) for v in ks}
        for v in range(n + 1):
            anc, w = {0, v}, v
            while w:
                w = parent[w]
                anc.add(w)
            assert set(torch.nonzero(mask_t[0, 0, v]).view(-1).tolist()) == anc and pos[v] == len(anc) - 1, v
        assert sorted(max(r) for r in rows) == [v for v in range(1, n + 1) if not kids[v]]
        assert len(S["loc"]) == n + 1 and None not in S["loc"]
        for v, (r, d) in enumerate(S["loc"]):
            chain, w = [], v
            while w:
                chain.append(w)
                w = parent[w]
            assert rows[r][: d + 1] == [0] + chain[::-1], v
        t2d = {t: i for i, t in enumerate(S["ids"].tolist())}
        score = {0: 0.0}
        for u in range(n + 1):
            if kids[u]:
                assert len(kids[u]) == k and len({toks[v] for v in kids[u]}) == k, u
                lq, j = S["logq"], S["qrow"][u]
                for v in kids[u]:
                    score[v] = score[u] + float(lq[j, t2d[toks[v]]])
        assert {u for u in range(n + 1) if S["qrow"][u] >= 0} == {u for u in range(n + 1) if kids[u]}
        assert all(S["spine_list"][v] == ([v] + S["spine_list"][kids[v][0]] if kids[v] else [v]) for v in range(n + 1))
        level = collections.defaultdict(list)
        for v in range(1, n + 1):
            level[pos[v]].append(v)
        spine = kids[0][:spines]  # first-drawn children down from the first `spines` root children
        for d in range(1, depth + extra + 2):
            if d > depth + extra:
                want = set()
            elif d > depth:  # extra levels: only the first spine goes on
                want = {spine[0]}
            else:
                rest = [v for v in level[d] if v not in spine]
                want = set(spine) | set(sorted(rest, key=lambda v: -score[v])[: k - len(spine)])
            assert {v for v in level[d] if kids[v]} == want, (spines, extra, d)
            if d < depth:
                spine = [kids[u][0] for u in spine]
            elif d < depth + extra:
                spine = [kids[spine[0]][0]]
        print(f"sampled tree (spines={spines}, extra={extra}): masks, positions, rows, (row, depth) map, distinct "
              f"siblings, expanded nodes per level: ok")
    # adapter: per-node target logits near q (random at leaves), gathered by rows as tree_decoding does
    node_logits = torch.randn(n + 1, config.vocab_size, generator=g) * 3
    for u in (u for u in range(n + 1) if kids[u]):
        q = torch.zeros(config.vocab_size)
        q[S["ids"]] = torch.softmax(S["logq"][S["qrow"][u]], dim=-1)
        node_logits[u] = torch.log(q + 1e-6) + 0.3 * torch.randn(config.vocab_size, generator=g)
    logits = node_logits[rows_t]
    walkers = [("rrsw", st.rrsw), ("hsd", st.hsd)] + ([("hsd_fused", st.hsd_fused)] if st.triton else [])
    for name, walk in walkers:
        ev = st.make_evaluate_posterior(walk, shadow=3 if name != "rrsw" else 0,
                                        shadow_hsd=st.hsd_fused if name == "hsd_fused" else None)
        acc = []
        st.STATS.clear()
        for s in range(100):
            random.seed(s)
            best, a, bonus = ev(logits, None, lp, hsd=True)
            row = rows[int(best)]
            assert all(parent[row[d]] == row[d - 1] for d in range(1, a + 1)), (s, row, a)
            assert abs(float(bonus.sum()) - 1) < 1e-9 and float(bonus.min()) >= 0
            acc.append(a)
        assert len(st.STATS) == 100 and all(r["spine"] == depth + 1 + extra for r in st.STATS)
        note = ""
        if name != "rrsw":
            note = (f"; shadow walks on the same trees: rrsw {np.mean([r['rrsw'] for r in st.STATS]):.2f}, "
                    f"hsd {np.mean([r['hsd'] for r in st.STATS]):.2f}")
        print(f"evaluate_posterior({name}) on the spines={spines}, extra={extra} tree: rows consistent with parents, "
              f"bonus distribution valid; mean accepted {np.mean(acc):.2f} with p ~ q{note}")
    if st.triton:  # the fused walker against hsd on this tree, same coins
        view = st.TreeView(logits, lp)
        worst = same_walks(view, range(300))
        print(f"hsd_fused == hsd on the spines={spines}, extra={extra} tree over 300 coin streams: same paths, "
              f"max |bonus difference| {worst:.1e}")


def same_walks(view, seeds):
    """Walk hsd and hsd_fused on view with the same coin stream per seed: assert the same accepted
    nodes, return the largest bonus-distribution difference."""
    worst = 0.0
    for s in seeds:
        a, b = random.Random(s), random.Random(s)
        pa, ra = st.hsd(view, a.random)
        pb, rb = st.hsd_fused(view, b.random)
        assert pa == pb and a.getstate() == b.getstate(), (s, pa, pb)
        worst = max(worst, float((ra - rb).abs().max()))
    assert worst < 1e-9, worst
    return worst


def check_fused():
    """F: hsd_fused against hsd on the toy environments of part A, plain and with extra levels."""
    for V, depth, W, alpha, mix, seed in [(3, 3, 2, 0.5, 0.0, 0), (5, 2, 2, 0.3, 0.5, 2), (4, 3, 2, 0.5, 0.3, 3)]:
        env = Env(V, seed, alpha, alpha, 0.0, mix)
        for spines, extra in ((0, 0), (1, 0), (1, 2), (2, 3)):
            rng, worst, t0 = np.random.default_rng(seed), 0.0, time.time()
            for t in range(150):
                view = EnvView(env, sample_tree(env, depth, W, rng, spines, extra))
                worst = max(worst, same_walks(view, range(t * 4, t * 4 + 4)))
            print(f"F V={V} depth={depth} W={W} spines={spines} extra={extra}: hsd_fused == hsd on 150 trees x 4 coin "
                  f"streams, max |bonus difference| {worst:.1e} | {time.time() - t0:.0f}s", flush=True)


def check_children(M=400000):
    q = np.array([0.5, 0.3, 0.15, 0.05])
    torch.manual_seed(0)
    logq = torch.tensor(np.log(q), dtype=torch.float32).expand(M, len(q)).contiguous()
    idx, lq = st.children(logq, 2, True)
    assert torch.equal(lq, logq.gather(-1, idx))
    got = collections.Counter(map(tuple, idx.tolist()))
    law = wor_sequences(q, 2)
    z = [abs(got.get(s, 0) / M - pr) / np.sqrt(pr * (1 - pr) / M) for s, pr in law]
    assert set(got) <= {s for s, _ in law}
    print(f"children(sample=True): {M} draws of 2 of 4 tokens, max |z| {max(z):.2f} over {len(law)} ordered pairs")
    assert st.children(logq[:1], 2, False)[0].tolist() == [[0, 1]]


if __name__ == "__main__":
    M = int(sys.argv[sys.argv.index("--mc") + 1]) if "--mc" in sys.argv else 100000
    parts = [a for a in sys.argv[1:] if a in ("A", "A1", "A2", "AX", "B", "C", "F")] or ["C", "B", "F", "A", "A1",
                                                                                          "A2", "AX"]
    for part in parts:  # A1 / A2: part A with 1 / 2 of the 2 expansions per level on first-child paths
        {"A": lambda: check_walkers(M), "A1": lambda: check_walkers(M, spines=1),
         "A2": lambda: check_walkers(M, spines=2), "AX": lambda: check_walkers(M, spines=1, extra=2),
         "B": check_builder, "C": check_children, "F": check_fused}[part]()
