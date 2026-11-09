"""Exact check of HSD (arXiv 2601.05724) vs block verification (arXiv 2403.10444).

Verifiers, all for one chain draft X_1..X_g sampled from q:
  tokenwise  Leviathan et al.
  block      Sun et al. Algorithm 2, same math as sglang's BLOCK_VERIFICATION kernel branch
  hsd_paper  HSD Defs 4-6, Eqs 16-20, computed literally from m(X) and r*(X)
  hsd_code   paper Listing 2 / repo _speculative_sampling(backward=True), multidraft=1

Losslessness: run the verifier step after step from the root context and compare
the law of the first L output tokens with the target joint (exact enumeration).
"""
import itertools

import numpy as np


class Env:
    """Target p(.|ctx) and draft q(.|ctx) for every context, fixed by the seed."""

    def __init__(self, V, seed, alpha_p=0.5, alpha_q=0.5, p_zero=0.0, mix=0.0):
        self.V, self.seed = V, seed
        self.alpha_p, self.alpha_q, self.p_zero, self.mix = alpha_p, alpha_q, p_zero, mix
        self.cache = {}

    def pq(self, ctx):
        if ctx not in self.cache:
            rng = np.random.default_rng([self.seed, len(ctx), *ctx])
            p = rng.dirichlet([self.alpha_p] * self.V)
            q = rng.dirichlet([self.alpha_q] * self.V)
            if self.p_zero > 0:
                zero = rng.random(self.V) < self.p_zero
                zero[rng.integers(self.V)] = False
                p = np.where(zero, 0.0, p)
                p /= p.sum()
            q = (1 - self.mix) * q + self.mix * p
            self.cache[ctx] = (p, q)
        return self.cache[ctx]


def scan(h, res):
    """Backward scan: tau = max{k : eta_k <= h_k}, h[0] = 1. Returns [(tau, prob, residual)]."""
    out, none_above = [], 1.0
    for k in range(len(h) - 1, -1, -1):
        pk = none_above * h[k]
        if pk > 0:
            out.append((k, pk, res[k]))
        none_above *= 1 - h[k]
    return out


def _norm(v):
    s = v.sum()
    return v / s if s > 0 else None


def tokenwise(P, Q, x):
    g, out, reach = len(x), [], 1.0
    for k in range(g):
        a = min(1.0, P[k, x[k]] / Q[k, x[k]])
        if reach * (1 - a) > 0:
            out.append((k, reach * (1 - a), _norm(np.maximum(P[k] - Q[k], 0.0))))
        reach *= a
    out.append((g, reach, P[g]))
    return out


def block_h(P, Q, x):
    g = len(x)
    rho = [1.0]
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


def hsd_paper_h(P, Q, x):
    g = len(x)
    a = [P[k, x[k]] / Q[k, x[k]] for k in range(g)]
    R = np.cumprod([1.0] + a)  # R[i] = r(X_{1:i})
    qj = np.cumprod([1.0] + [Q[k, x[k]] for k in range(g)])  # q(X_{1:i})

    def m_of(t):  # Def 4: argmax_{1<=i<t} r(X_{1:i}), or 0 if that max <= 1
        if t <= 1:
            return 0
        i = 1 + int(np.argmax(R[1:t]))
        return i if R[i] > 1 else 0

    def term(t):  # (r*(X_{1:t-1}, x) - 1) q(X_{1:t-1}, x) for every sibling x at position t
        m = m_of(t)
        r_mid = float(np.prod(a[m : t - 1]))  # r(X_{m+1:t-1})
        return min(R[m], 1.0) * r_mid * qj[t - 1] * P[t - 1] - qj[t - 1] * Q[t - 1]

    h, res = np.zeros(g + 1), [None] * (g + 1)
    h[0] = 1.0
    for t in range(0, g):
        d = term(t + 1)
        Dpq, Dqp = np.maximum(d, 0).sum(), np.maximum(-d, 0).sum()  # Eqs 17, 18
        res[t] = _norm(np.maximum(d, 0))  # Eq 20
        if t > 0:
            h[t] = Dpq / Dqp if Dqp > 0 else 1.0  # Eq 19
    m = m_of(g)
    rstar_g = min(R[m], 1.0) * float(np.prod(a[m:g]))  # Def 5 at t = g
    h[g], res[g] = min(rstar_g, 1.0), P[g]
    return h, res


def hsd_code_steps(P, Q, x):
    g = len(x)
    qi = np.array([Q[k, x[k]] for k in range(g)])
    pi = np.array([P[k, x[k]] for k in range(g)])
    q_cum = np.concatenate([[1.0], np.cumprod(qi)[:-1]])
    p_cum = np.concatenate([[1.0], np.cumprod(pi)[:-1]])
    prev_max = np.maximum.accumulate(np.maximum(p_cum / q_cum, 1.0))
    diffs = (p_cum / prev_max)[:, None] * P[:g] - q_cum[:, None] * Q[:g]
    p_plus, p_minus = np.clip(diffs, 0, None), np.clip(-diffs, 0, None)
    p_primes = p_plus / np.maximum(p_plus.sum(-1, keepdims=True), p_minus.sum(-1, keepdims=True))
    h_mid = p_primes.sum(-1)  # 1 - step_back_probs
    full = min(1.0, float(np.prod(pi / qi)))  # is_accepted[:, -1]: uncapped joint ratio
    out, none_above = [(g, full, P[g])], 1.0 - full
    for k in range(g - 1, -1, -1):
        pk = none_above * h_mid[k]
        if pk > 0:
            out.append((k, pk, _norm(p_primes[k])))
        none_above *= 1 - h_mid[k]
    return out


VERIFIERS = {
    "tokenwise": tokenwise,
    "block": lambda P, Q, x: scan(*block_h(P, Q, x)),
    "hsd_paper": lambda P, Q, x: scan(*hsd_paper_h(P, Q, x)),
    "hsd_code": hsd_code_steps,
}


def drafts(env, ctx, g):
    """Every chain draft from ctx with its q-probability and the per-position P, Q."""
    for x in itertools.product(range(env.V), repeat=g):
        P, Q, qx, c = [], [], 1.0, ctx
        for k in range(g):
            p, q = env.pq(c)
            P.append(p)
            Q.append(q)
            qx *= q[x[k]]
            c = c + (x[k],)
        if qx > 0:
            P.append(env.pq(c)[0])
            yield x, qx, np.array(P), np.array(Q)


def step_dist(env, ctx, g, verify):
    """Law of one verify step's output X_{1:tau} + (Y,)."""
    d = {}
    for x, qx, P, Q in drafts(env, ctx, g):
        for tau, pt, res in verify(P, Q, x):
            for y in np.nonzero(res)[0]:
                key = x[:tau] + (int(y),)
                d[key] = d.get(key, 0.0) + qx * pt * res[y]
    return d


def horizon_dist(env, ctx, L, g, verify, memo):
    """Law of the first L tokens when verify steps are chained from ctx."""
    if (ctx, L) not in memo:
        out = {}
        for seq, pr in step_dist(env, ctx, g, verify).items():
            if len(seq) >= L:
                out[seq[:L]] = out.get(seq[:L], 0.0) + pr
            else:
                for rest, pr2 in horizon_dist(env, ctx + seq, L - len(seq), g, verify, memo).items():
                    out[seq + rest] = out.get(seq + rest, 0.0) + pr * pr2
        memo[(ctx, L)] = out
    return memo[(ctx, L)]


def target_joint(env, ctx, L):
    out = {}
    for s in itertools.product(range(env.V), repeat=L):
        pr, c = 1.0, ctx
        for t in s:
            pr *= env.pq(c)[0][t]
            c = c + (t,)
        if pr > 0:
            out[s] = pr
    return out


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def max_rule_gap(env, g, ctxs):
    """Largest |h| and residual gap between block_h and hsd_paper_h over all drafts."""
    dh = dres = 0.0
    for ctx in ctxs:
        for x, _, P, Q in drafts(env, ctx, g):
            (h1, r1), (h2, r2) = block_h(P, Q, x), hsd_paper_h(P, Q, x)
            dh = max(dh, float(np.max(np.abs(h1 - h2))))
            for a, b in zip(r1, r2):
                if a is not None and b is not None:
                    dres = max(dres, float(np.max(np.abs(a - b))))
                elif (a is None) != (b is None) and h1[r1.index(a) if a is not None else r2.index(b)] > 0:
                    dres = float("inf")
    return dh, dres


CONFIGS = [  # (V, g, L, alpha_p, alpha_q, p_zero, mix)
    (3, 2, 3, 0.5, 0.5, 0.0, 0.0),
    (3, 2, 3, 0.2, 0.2, 0.3, 0.5),
    (4, 3, 4, 0.5, 0.5, 0.0, 0.0),
    (4, 3, 4, 1.0, 1.0, 0.0, 0.6),
    (4, 3, 4, 0.2, 0.5, 0.3, 0.3),
    (3, 4, 5, 0.5, 0.5, 0.0, 0.0),
    (3, 4, 5, 0.3, 0.3, 0.3, 0.6),
]


def main(seeds=range(20)):
    names = list(VERIFIERS)
    print("cfg (V,g,L,ap,aq,p0,mix) | max|dh| max|dres| block vs hsd_paper"
          " | max TV vs target per verifier | mean tokens/step")
    for cfg in CONFIGS:
        V, g, L, ap, aq, p0, mix = cfg
        worst = {n: 0.0 for n in names}
        mean_len = {n: 0.0 for n in names}
        gap_h = gap_r = 0.0
        for seed in seeds:
            env = Env(V, seed, ap, aq, p0, mix)
            dh, dr = max_rule_gap(env, g, [(), (0,), (1, 2)])
            gap_h, gap_r = max(gap_h, dh), max(gap_r, dr)
            target = target_joint(env, (), L)
            for n in names:
                got = horizon_dist(env, (), L, g, VERIFIERS[n], {})
                worst[n] = max(worst[n], tv(got, target))
                sd = step_dist(env, (), g, VERIFIERS[n])
                mean_len[n] += sum(pr * len(s) for s, pr in sd.items()) / len(seeds)
        print(
            cfg,
            f"| {gap_h:.1e} {gap_r:.1e} |",
            " ".join(f"{n}={worst[n]:.1e}" for n in names),
            "|",
            " ".join(f"{n}={mean_len[n]:.4f}" for n in names),
            flush=True,
        )


if __name__ == "__main__":
    main()
