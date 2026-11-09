"""Exact check of tree verification when each expanded node's children are drawn from q WITHOUT
replacement (Gumbel top-W order = sequential sampling order), as the GPU bench will do.

Tree shape mirrors the bench: the root gets W children; at every deeper level only the E nodes with
the highest path log-q are expanded (W children each); nothing is pruned afterwards.
  rrsw_tree     token-level recursive rejection sampling without replacement: trial i proposes
                c_i ~ q_i = q with c_1..c_{i-1} removed; reject -> target becomes norm((R - q_i)_+)
  hsd_tree_wor  Algorithm 3 on the tree: verify c_i and then first children down to a leaf with block
                verification (proposal q_i at the first position); stop at a node -> continue with its
                next untried child
Reference: the target joint, by chaining verify steps (as in verify_hsd_exact.py).
"""
import itertools
import sys

import numpy as np

from verify_alg3_exact import Shifted, _emit, path_pq
from verify_hsd_exact import Env, _norm, scan, target_joint, tv


def wor_sequences(q, W):
    """Ordered W-tuples of distinct tokens with their sequential without-replacement probability."""
    out = []
    for seq in itertools.permutations(range(len(q)), W):
        pr, left = 1.0, 1.0
        for t in seq:
            if left <= 0:
                pr = 0.0
                break
            pr *= q[t] / left
            left -= q[t]
        if pr > 0:
            out.append((seq, pr))
    return out


def trees(env, depth, W, E, spines=0, extra=0):
    """All trees with their probability. A tree is {prefix tuple: children tuple}.
    spines=m reserves m of the E expansions per level for the paths of first-drawn children under the
    first m root children; the rest go to the highest path log-q as without it. extra=e adds e levels
    below where only the first of those paths goes on (needs spines >= 1)."""
    assert extra == 0 or spines >= 1
    results = [({}, 1.0)]
    frontier = [()]
    for level in range(depth + extra):
        new_results = []
        for tree, pr in results:
            nodes = frontier if level == 0 else None
            if level >= depth:
                nodes = spine_nodes(tree, level, 1)
            elif level > 0:
                cand = [p for p in tree_nodes(tree) if len(p) == level]
                spine = spine_nodes(tree, level, spines)
                rest = [p for p in cand if p not in spine]
                # expand the other E - m deepest-level nodes with the highest path log-q (ties: first in order)
                scores = [path_logq(env, p) for p in rest]
                order = sorted(range(len(rest)), key=lambda i: -scores[i])
                nodes = spine + [rest[i] for i in order[: E - len(spine)]]
            combos = [wor_sequences(env.pq(n)[1], W) for n in nodes]
            for choice in itertools.product(*combos):
                t2 = dict(tree)
                p2 = pr
                for n, (seq, ps) in zip(nodes, choice):
                    t2[n] = seq
                    p2 *= ps
                new_results.append((t2, p2))
        results = new_results
    return results


def tree_nodes(tree):
    out = []
    for parent, kids in tree.items():
        out += [parent + (k,) for k in kids]
    return out


def spine_nodes(tree, level, m):
    """The level-`level` nodes on the paths of first-drawn children under the first m root children."""
    s = [(c,) for c in tree[()][:m]]
    for _ in range(level - 1):
        s = [p + (tree[p][0],) for p in s]
    return s


def path_logq(env, path):
    s = 0.0
    for i, t in enumerate(path):
        s += np.log(env.pq(path[:i])[1][t])
    return s


def restricted(q, used):
    r = q.copy()
    r[list(used)] = 0.0
    return r / r.sum()


def rrsw_tree(env, tree):
    out = {}

    def go(prefix, target, i, used):
        kids = tree.get(prefix, ())
        if i == len(kids):
            return [(prefix, target, 1.0)]
        q = restricted(env.pq(prefix)[1], used)
        c = kids[i]
        a = min(1.0, target[c] / q[c])
        res = []
        if a > 0:
            nxt = prefix + (c,)
            res += [(s, d, a * w) for s, d, w in go(nxt, env.pq(nxt)[0], 0, ())]
        if a < 1:
            r = np.maximum(target - q, 0.0)
            res += [(s, d, (1 - a) * w) for s, d, w in go(prefix, r / r.sum(), i + 1, used + (c,))]
        return res

    for seq, dist, w in go((), env.pq(())[0], 0, ()):
        _emit(out, seq, dist, w)
    return out


def hsd_tree_wor(env, tree):
    out = {}

    def go(prefix, target, i, used, prob):
        kids = tree.get(prefix, ())
        if i == len(kids):
            _emit(out, prefix, target, prob)
            return
        toks = [kids[i]]
        node = prefix + (kids[i],)
        while tree.get(node):
            toks.append(tree[node][0])
            node = node + (tree[node][0],)
        toks = tuple(toks)
        P, Q = path_pq(env, prefix, toks, target)
        Q[0] = restricted(Q[0], used)  # first position: proposal without the tried siblings
        g = len(toks)
        rho = [1.0]
        for k in range(g):
            rho.append(min(rho[-1] * P[k, toks[k]] / Q[k, toks[k]], 1.0))
        h, res = np.zeros(g + 1), [None] * (g + 1)
        h[0] = 1.0
        for k in range(g):
            r = np.maximum(rho[k] * P[k] - Q[k], 0.0)
            S = r.sum()
            res[k] = _norm(r)
            if k > 0:
                h[k] = S / (S + 1.0 - rho[k]) if rho[k] < 1.0 else 1.0
        h[g], res[g] = rho[g], P[g]
        for j, pj, r in scan(h, res):
            if j == g:
                _emit(out, prefix + toks, P[-1], prob * pj)
            elif j == 0:
                go(prefix, r, i + 1, used + (kids[i],), prob * pj)
            else:  # stopped below: the path went through that node's first child
                v = prefix + toks[:j]
                go(v, r, 1, (toks[j],), prob * pj)

    go((), env.pq(())[0], 0, (), 1.0)
    return out


def topk_tree(env, depth, W, E):
    tree = {(): tuple(np.argsort(-env.pq(())[1], kind="stable")[:W].tolist())}
    for level in range(1, depth):
        cand = [p for p in tree_nodes(tree) if len(p) == level]
        cand = sorted(cand, key=lambda p: -path_logq(env, p))[:E]
        for n in cand:
            tree[n] = tuple(np.argsort(-env.pq(n)[1], kind="stable")[:W].tolist())
    return tree


def target_only_len(env, tree):
    total = 1.0
    for n in tree_nodes(tree):
        pr = 1.0
        for i, t in enumerate(n):
            pr *= env.pq(n[:i])[0][t]
        total += pr
    return total


def target_only(env, tree):
    """Deterministic children, q = 1 on the drafted token: accept child c with prob R(c)."""
    out = {}

    def go(prefix, target, i, prob):
        kids = tree.get(prefix, ())
        if i == len(kids):
            _emit(out, prefix, target, prob)
            return
        c = kids[i]
        a = target[c]
        if a > 0:
            nxt = prefix + (c,)
            go(nxt, env.pq(nxt)[0], 0, prob * a)
        if a < 1:
            r = target.copy()
            r[c] = 0.0
            go(prefix, r / r.sum(), i + 1, prob * (1 - a))

    go((), env.pq(())[0], 0, 1.0)
    return out


def step_law(env, ctx, depth, W, E, verify, sampled=True, spines=0, extra=0):
    senv = Shifted(env, ctx)
    d = {}
    pairs = trees(senv, depth, W, E, spines, extra) if sampled else [(topk_tree(senv, depth, W, E), 1.0)]
    for tree, pr in pairs:
        for seq, ps in verify(senv, tree).items():
            d[seq] = d.get(seq, 0.0) + pr * ps
    return d


def horizon(env, ctx, L, depth, W, E, verify, sampled, memo, spines=0, extra=0):
    if (ctx, L) not in memo:
        out = {}
        for seq, pr in step_law(env, ctx, depth, W, E, verify, sampled, spines, extra).items():
            if len(seq) >= L:
                out[seq[:L]] = out.get(seq[:L], 0.0) + pr
            else:
                for rest, pr2 in horizon(env, ctx + seq, L - len(seq), depth, W, E, verify, sampled, memo,
                                         spines, extra).items():
                    out[seq + rest] = out.get(seq + rest, 0.0) + pr * pr2
        memo[(ctx, L)] = out
    return memo[(ctx, L)]


def mean_len(d):
    return sum(pr * len(s) for s, pr in d.items())


if __name__ == "__main__":
    cases = [(3, 2, 2, 2), (4, 2, 2, 2)]  # (V, depth, W, E)
    # spines=m: m expansions per level continue first-child paths (default 0: all by path log q)
    spines = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("spines=")), 0))
    # extra=e: e more levels where only the first of those paths goes on
    extra = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("extra=")), 0))
    if len(sys.argv) > 1:
        cases.append((3, 3, 2, 2))  # level-2 frontier keeps 2 of 4 nodes
    if extra:  # the extra levels start right below `depth`, so the two depth-2 cases already reach them
        cases = cases[:2]
    elif spines:  # the rules differ only from the second level of expansions on
        cases = cases[2:]
    for V, depth, W, E in cases:
        L = depth + extra + 1  # the law of every token one step can emit
        for ap, mix in [(0.5, 0.0), (1.0, 0.6), (0.3, 0.5)]:
            worst = {"rrsw": 0.0, "hsd_wor": 0.0, "topk_target_only": 0.0}
            lens = dict.fromkeys(worst, 0.0)
            seeds = range(4)
            for seed in seeds:
                env = Env(V, seed, ap, ap, 0.0, mix)
                target = target_joint(env, (), L)
                for name, f, sampled in (("rrsw", rrsw_tree, True), ("hsd_wor", hsd_tree_wor, True),
                                         ("topk_target_only", target_only, False)):
                    ex = extra if sampled else 0
                    worst[name] = max(worst[name], tv(horizon(env, (), L, depth, W, E, f, sampled, {}, spines, ex),
                                                      target))
                    lens[name] += mean_len(step_law(env, (), depth, W, E, f, sampled, spines, ex)) / len(seeds)
            print(f"V={V} depth={depth} W={W} E={E} spines={spines} extra={extra} alpha={ap} mix={mix} | max TV vs "
                  "target: " + " ".join(f"{k}={v:.1e}" for k, v in worst.items()) + " | tokens/step: "
                  + " ".join(f"{k}={v:.4f}" for k, v in lens.items()), flush=True)
