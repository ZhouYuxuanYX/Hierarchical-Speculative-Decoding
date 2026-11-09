"""Exact check of HSD Algorithm 3 (Appendix G, multi-draft) against token-level RRS.

K drafts of length g are sampled i.i.d. from q (RRS with replacement, as Appendix G says).
  tokenwise / block   single draft (K=1) references from verify_hsd_exact.py
  rrs        token-level recursive rejection sampling: at each position try the drafts that
             share the accepted prefix in index order; a rejection turns (p - q)_+ into the target
  alg3       Algorithm 3: verify one draft at a time with Eq 19 (capped, same as block
             verification), ratios restarting at the accepted prefix; after a rejection P*_res
             becomes the target at the next position for the later drafts (lines 28-29)
  alg3_uncapped  alg3 with the repo's full-draft test min(r(X_{tau+1:g}), 1)
Losslessness is checked as in verify_hsd_exact.py: chain verify steps from the root and
compare the law of the first L tokens with the target joint.
"""
import itertools

import numpy as np

from verify_hsd_exact import VERIFIERS, Env, _norm, block_h, scan, target_joint, tv


class Shifted:
    def __init__(self, env, ctx):
        self.env, self.ctx, self.V = env, ctx, env.V

    def pq(self, c):
        return self.env.pq(self.ctx + c)


def _emit(out, seq, dist, pr):
    for y in np.nonzero(dist)[0]:
        key = seq + (int(y),)
        out[key] = out.get(key, 0.0) + pr * dist[y]


def path_pq(env, prefix, suffix, first_target):
    P, Q, c = [first_target], [], prefix
    for t in suffix:
        Q.append(env.pq(c)[1])
        c = c + (t,)
        P.append(env.pq(c)[0])
    return np.array(P), np.array(Q)


def rrs(env, drafts):
    K, g, out = len(drafts), len(drafts[0]), {}

    def go(prefix, target, cands, prob):
        t = len(prefix)
        if t == g or not cands:
            _emit(out, prefix, target, prob)
            return
        q = env.pq(prefix)[1]
        x = drafts[cands[0]][t]
        a = min(1.0, target[x] / q[x])
        if a > 0:
            nxt = prefix + (x,)
            go(nxt, env.pq(nxt)[0], [j for j in range(K) if drafts[j][: t + 1] == nxt], prob * a)
        if a < 1:
            r = np.maximum(target - q, 0.0)
            go(prefix, r / r.sum(), cands[1:], prob * (1 - a))

    go((), env.pq(())[0], list(range(K)), 1.0)
    return out


def alg3(env, drafts, uncapped_full=False):
    K, out = len(drafts), {}

    def go(prefix, target, k0, prob):
        tau = len(prefix)
        k = next((j for j in range(k0, K) if drafts[j][:tau] == prefix), None)
        if k is None:
            _emit(out, prefix, target, prob)
            return
        suffix = drafts[k][tau:]
        P, Q = path_pq(env, prefix, suffix, target)
        h, res = block_h(P, Q, suffix)
        if uncapped_full:
            h = h.copy()
            h[-1] = min(1.0, float(np.prod([P[i, s] / Q[i, s] for i, s in enumerate(suffix)])))
        for j, pj, r in scan(h, res):
            if j == len(suffix):
                _emit(out, prefix + suffix, P[-1], prob * pj)
            else:
                go(prefix + suffix[:j], r, k + 1, prob * pj)

    go((), env.pq(())[0], 0, 1.0)
    return out



def block_h_from(P, Q, x, rho0):
    """block_h with the capped prefix ratio starting at rho0 instead of 1."""
    g, rho = len(x), [rho0]
    for k in range(g):
        rho.append(min(rho[-1] * P[k, x[k]] / Q[k, x[k]], 1.0))
    h, res = np.zeros(g + 1), [None] * (g + 1)
    h[0] = 1.0
    for k in range(g):
        r = np.maximum(rho[k] * P[k] - Q[k], 0.0)
        S = r.sum()
        res[k] = _norm(r)
        if k > 0:
            h[k] = S / (S + 1.0 - rho[k]) if rho[k] < 1.0 else 1.0
    h[g], res[g] = rho[g], P[g]
    return h, res


def alg3_repo(env, drafts, uncapped_full=True):
    """Repo multi-draft path (parallel=True, utils.py:5317-5378): a later draft starts the
    ratio from the accepted prefix's joint ratio (lines 5333, 5343), not from 1."""
    K, out = len(drafts), {}

    def go(prefix, target, k0, prob, R):
        tau = len(prefix)
        k = next((j for j in range(k0, K) if drafts[j][:tau] == prefix), None)
        if k is None:
            _emit(out, prefix, target, prob)
            return
        suffix = drafts[k][tau:]
        P, Q = path_pq(env, prefix, suffix, target)
        a = [P[i, s] / Q[i, s] for i, s in enumerate(suffix)]
        h, res = block_h_from(P, Q, suffix, min(R, 1.0))
        if uncapped_full:
            h = h.copy()
            h[-1] = min(1.0, float(np.prod(a)))
        for j, pj, r in scan(h, res):
            if j == len(suffix):
                _emit(out, prefix + suffix, P[-1], prob * pj)
            else:
                go(prefix + suffix[:j], r, k + 1, prob * pj, R * float(np.prod(a[:j])))

    go((), env.pq(())[0], 0, 1.0, 1.0)
    return out

MULTI = {
    "rrs": rrs,
    "alg3": alg3,
    "alg3_uncapped": lambda env, drafts: alg3(env, drafts, uncapped_full=True),
    "repo_carry_only": lambda env, drafts: alg3_repo(env, drafts, uncapped_full=False),
    "repo": alg3_repo,
}


def single_drafts(env, g):
    for x in itertools.product(range(env.V), repeat=g):
        pr, c = 1.0, ()
        for t in x:
            pr *= env.pq(c)[1][t]
            c = c + (t,)
        if pr > 0:
            yield x, pr


def step_dist_multi(env, ctx, g, K, verify):
    senv = Shifted(env, ctx)
    singles = list(single_drafts(senv, g))
    d = {}
    for combo in itertools.product(singles, repeat=K):
        pr = float(np.prod([c[1] for c in combo]))
        for seq, ps in verify(senv, [c[0] for c in combo]).items():
            key = ctx[:0] + seq
            d[key] = d.get(key, 0.0) + pr * ps
    return d


def horizon_multi(env, ctx, L, g, K, verify, memo):
    if (ctx, L) not in memo:
        out = {}
        for seq, pr in step_dist_multi(env, ctx, g, K, verify).items():
            if len(seq) >= L:
                out[seq[:L]] = out.get(seq[:L], 0.0) + pr
            else:
                for rest, pr2 in horizon_multi(env, ctx + seq, L - len(seq), g, K, verify, memo).items():
                    out[seq + rest] = out.get(seq + rest, 0.0) + pr * pr2
        memo[(ctx, L)] = out
    return memo[(ctx, L)]


def mean_len(d):
    return sum(pr * len(s) for s, pr in d.items())


CONFIGS = [  # (V, g, K, alpha_p, alpha_q, p_zero, mix)
    (3, 2, 2, 0.5, 0.5, 0.0, 0.0),
    (3, 2, 2, 1.0, 1.0, 0.0, 0.6),
    (3, 2, 3, 0.5, 0.5, 0.0, 0.0),
    (3, 2, 3, 0.3, 0.3, 0.3, 0.5),
    (4, 2, 2, 0.5, 0.5, 0.0, 0.3),
    (3, 3, 2, 0.5, 0.5, 0.0, 0.0),
    (3, 3, 2, 1.0, 1.0, 0.0, 0.6),
]


def main(seeds=range(10)):
    from verify_hsd_exact import step_dist

    # K=1 sanity: rrs must equal tokenwise and alg3 must equal block.
    env = Env(3, 0)
    for name, ref in (("rrs", "tokenwise"), ("alg3", "block")):
        gap = tv(step_dist_multi(env, (), 2, 1, MULTI[name]), step_dist(env, (), 2, VERIFIERS[ref]))
        print(f"K=1 sanity: TV({name}, {ref}) = {gap:.1e}")
    names = list(MULTI)
    print("cfg (V,g,K,ap,aq,p0,mix) | max TV vs target (L=g+1) | mean tokens/step:"
          " tokenwise(K=1) block(K=1) then K-draft verifiers")
    for cfg in CONFIGS:
        V, g, K, ap, aq, p0, mix = cfg
        worst = {n: 0.0 for n in names}
        lens = {n: 0.0 for n in ["tokenwise", "block"] + names}
        for seed in seeds:
            env = Env(V, seed, ap, aq, p0, mix)
            target = target_joint(env, (), g + 1)
            for n in names:
                got = horizon_multi(env, (), g + 1, g, K, MULTI[n], {})
                worst[n] = max(worst[n], tv(got, target))
                lens[n] += mean_len(step_dist_multi(env, (), g, K, MULTI[n])) / len(seeds)
            for n in ("tokenwise", "block"):
                lens[n] += mean_len(step_dist(env, (), g, VERIFIERS[n])) / len(seeds)
        print(cfg, "|", " ".join(f"{n}={worst[n]:.1e}" for n in names), "|",
              " ".join(f"{n}={lens[n]:.4f}" for n in lens), flush=True)


if __name__ == "__main__":
    main()
