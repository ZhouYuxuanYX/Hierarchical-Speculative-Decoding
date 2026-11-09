"""Sampled EAGLE-3 draft trees and two lossless verifiers for them, as drop-in replacements for
eagle.model.cnets.Model.topK_genrate and eagle.model.utils.evaluate_posterior.

Tree: EAGLE's shape before its final pruning. The root gets k children; at each of `depth` levels
the k newest nodes with the highest path log q get k children each (spines=m reserves m of those k
for the paths of first-drawn children under the first m root children), so there are
k + depth * k^2 nodes. With sample=True the children of a node are k draws from the draft q without
replacement (Gumbel top-k, sibling order = draw order) instead of its top-k. Nothing is pruned:
dropping a node because of its own score would bias the draws, while choosing which nodes to expand
does not.

Verifiers walk the tree with the target p from the tree forward:
  rrsw  token-level recursive rejection sampling without replacement: the i-th child c of a node is
        accepted with min(1, R(c) / q_i(c)), q_i = q without the tried siblings; on rejection the
        target becomes norm((R - q_i)_+)
  hsd   HSD Algorithm 3 on the tree: block verification of the path child -> first children down
        to a leaf, with proposal q_i at the first position; a stop at a node continues with that
        node's next untried child under the residual of the stop position
  hsd_fused  the same walk with each path's tensor work in one Triton kernel (needs Triton)
test_sampled_tree.py checks the walkers against the exact laws in verify_sampled_tree_wor.py.
"""
import random
import time

import torch

try:  # only the fused HSD walker needs Triton
    import triton
    import triton.language as tl
except ImportError:
    triton = None

SIDE = {}  # the last tree built by topK_genrate, read by the verifiers
STATS = []  # per verification step, filled by make_evaluate_posterior
_SHADOW = random.Random(0)  # coins of the extra walks, so the generation's own random stream is untouched


def children(logq, k, sample):
    """k children for each row of log q: its top-k, or k draws without replacement in draw order."""
    if sample:
        keys = logq - torch.empty_like(logq).exponential_().log()  # log q + Gumbel noise
        idx = torch.topk(keys, k, dim=-1).indices
        return idx, logq.gather(-1, idx)
    top = torch.topk(logq, k, dim=-1)
    return top.indices, top.values


def _vocab_map(layer, dev):
    """Draft vocab -> target vocab ids (arange plus EAGLE's d2t offsets), built once per layer."""
    ids = getattr(layer, "_st_vocab_map", None)
    if ids is None or ids.device != dev:
        ids = torch.arange(layer.config.draft_vocab_size, device=dev)
        if layer.config.vocab_size != layer.config.draft_vocab_size:
            ids = ids + layer.d2t
        layer._st_vocab_map = ids
    return ids


@torch.no_grad()
def topK_genrate(self, hidden_states, input_ids, head, logits_processor, sample=True, spines=0, extra=0):
    """eagle.model.cnets.Model.topK_genrate with children() picking the children and no final
    pruning; bind it to the EAGLE layer with types.MethodType. Node 0 is the token just sampled and
    node 1 + f is entry f of EAGLE's flattened candidate list, so siblings are numbered in draw order.
    spines=m gives m of the k expansions per level to the first-drawn child of the nodes on the paths
    of first children under the first m root children, so those paths reach the last level; the
    other k - m go to the highest path log q as before (m = k: every expanded node's first child).
    extra=e adds e levels below the last where only the first spine goes on (its node's k children are
    drawn as usual), so that path holds depth + 1 + e draft tokens; needs spines >= 1.
    The shape of the tree follows from the nodes expanded per level, so the masks, rows and the
    verifiers' tables are built from lists after a single device-to-host copy."""
    assert extra == 0 or spines >= 1, "extra levels continue the first spine"
    k, depth, dev = self.top_k, self.depth, hidden_states.device
    input_ids = input_ids.to(dev)
    sample_token, input_ids = input_ids[:, -1], input_ids[:, 1:]
    len_posi = input_ids.shape[1]
    self.reset()
    if getattr(self, "stable_kv", None) is not None:
        kv_len = self.stable_kv[0][0].shape[2]
        out_hidden, past = self(hidden_states, input_ids=input_ids[:, kv_len:], past_key_values=self.stable_kv,
                                use_cache=True)
    else:
        out_hidden, past = self(hidden_states, input_ids=input_ids, use_cache=True)
    self.stable_kv = past
    last_hidden = out_hidden[:, -1]
    same_vocab = self.config.vocab_size == self.config.draft_vocab_size
    vocab_map = _vocab_map(self, dev)
    to_target = (lambda t: t) if same_vocab else (lambda t: vocab_map[t])

    # float32 log q: the verifiers use exactly these values as the proposal
    logq = [self.logsoftmax(self.lm_head(self.norm(last_hidden)).float())]
    idx, lq = children(logq[0], k, sample)
    scores, ids = lq[0], to_target(idx)
    tokens, sels = [ids[0]], []  # sels[i]: which of the nodes drawn in iteration i are expanded next
    input_ids, input_hidden, tree_mask = ids, last_hidden[None].repeat(1, k, 1), self.tree_mask_init
    spine_sel = torch.arange(spines, device=dev) * k  # first children of frontier rows 0..spines-1
    levels = depth + extra
    for i in range(levels):
        self.tree_mask = tree_mask
        pos = self.position_ids if i < depth else self.position_ids[:1]  # extra levels: one node
        out_hidden, past = self(input_hidden, input_ids=input_ids, past_key_values=past,
                                position_ids=len_posi + pos, use_cache=True)
        len_posi += 1
        logq.append(self.logsoftmax(self.lm_head(self.norm(out_hidden[0])).float()))
        idx, lq = children(logq[-1], k, sample)  # row j: children of the j-th node expanded here
        ids = to_target(idx).view(-1)
        tokens.append(ids)
        if i == levels - 1:  # the nodes drawn last are leaves: no selection, no next input
            break
        if i + 1 >= depth:  # into an extra level: only frontier row 0 (the first spine) goes on
            sel, block = spine_sel[:1], self.tree_mask_init[:, :, :1, :1]
        else:
            cu_scores = (lq + scores[:, None]).view(-1)
            # frontier rows 0..spines-1 are the spine nodes: keep them there with their first children
            if spines == 0:
                sel = torch.topk(cu_scores, k).indices
            elif spines < k:
                rest = cu_scores.index_fill(0, spine_sel, float("-inf"))
                sel = torch.cat([spine_sel, torch.topk(rest, k - spines).indices])
            else:
                sel = spine_sel
            scores, block = cu_scores[sel], self.tree_mask_init
        sels.append(sel)
        src = sel // k  # the frontier row each selected node hangs under
        input_hidden = out_hidden[:, src]
        input_ids = ids[sel][None]
        tree_mask = torch.cat((tree_mask[:, :, src], block), dim=3)

    draft_tokens = torch.cat([sample_token] + tokens)
    n1 = draft_tokens.shape[0]
    host = torch.cat([draft_tokens] + sels).tolist()  # the tree's only device-to-host copy
    # node ids expanded at each level (row j of logq[i + 1] belongs to expanded[i][j])
    expanded, at, first = [list(range(1, k + 1))], n1, 1 + k  # first: id of the first node drawn next
    for sel in sels:
        drawn = len(expanded[-1]) * k
        expanded.append([first + s for s in host[at:at + sel.shape[0]]])
        at, first = at + sel.shape[0], first + drawn
    parent = [0] * k + [u for level in expanded for u in level for _ in range(k)]  # parent[v - 1] of node v
    n = len(parent)
    path = [[0]]  # path[v]: the nodes from the root to v
    for u in parent:
        path.append(path[u] + [len(path)])
    width = max(map(len, path))
    inner = set(parent)
    rows = [s + [-1] * (width - len(s)) for v, s in enumerate(path) if v and v not in inner]
    if logits_processor is not None:  # EAGLE's row order when sampling
        rows.sort(key=lambda r: [x if x >= 0 else n + 5 for x in r])
    tree_mask = torch.zeros(n + 1, n + 1)
    tree_mask[[v for v, s in enumerate(path) for _ in s], [u for s in path for u in s]] = 1.0
    _record(self, draft_tokens, host[:n1], parent, rows, logq, expanded)
    return (draft_tokens[None], torch.tensor(rows, dtype=torch.long), tree_mask[None, None],
            torch.tensor([len(s) - 1 for s in path], device=dev))


def spines(kids):
    """For every node, the path of first-drawn children that starts at it (the node itself first)."""
    out = [None] * len(kids)
    for v in range(len(kids) - 1, -1, -1):  # children have larger ids than their parent
        out[v] = [v] + (out[kids[v][0]] if kids[v] else [])
    return out


def tables(kids, tokens, qrow):
    """The index tables of TreeView for a tree given as lists, on the CPU (for tests)."""
    sp = spines(kids)
    width = max(map(len, sp))
    return dict(kids=kids, tokens=tokens, qrow=qrow, spine_list=sp, spine_len=list(map(len, sp)),
                tok=torch.tensor(tokens), qr=torch.tensor([max(x, 0) for x in qrow]),
                spine=torch.tensor([s + [0] * (width - len(s)) for s in sp]), ar=torch.arange(width),
                zero=torch.zeros(1, dtype=torch.float64))


def _record(layer, draft_tokens, tokens, parent, rows, logq, expanded):
    """Store the tree for the verifiers: children in draw order, each node's (row, depth) in the
    gathered target logits, each expanded node's row in the stacked draft log q (row j of logq[i + 1]
    for the j-th node expanded at level i), and the first-child path below every node, as lists and
    as index tensors on the GPU (one host-to-device copy per tree). tokens is draft_tokens as a list."""
    n1, dev = len(tokens), draft_tokens.device
    kids = [[] for _ in range(n1)]
    for v, u in enumerate(parent, start=1):
        kids[u].append(v)
    qrow = [-1] * n1
    qrow[0] = 0
    r = 1  # rows of the stacked log q: the root's, then each level's expanded nodes in order
    for level in expanded:
        for j, u in enumerate(level):
            qrow[u] = r + j
        r += len(level)
    loc = [None] * n1
    for r, row in enumerate(rows):
        for d, v in enumerate(row):
            if v >= 0 and loc[v] is None:
                loc[v] = (r, d)
    sp = spines(kids)
    width = max(map(len, sp))
    idx = torch.tensor([r for r, _ in loc] + [d for _, d in loc] + [max(x, 0) for x in qrow]
                       + [v for s in sp for v in s + [0] * (width - len(s))]).to(dev)
    SIDE.update(tokens=tokens, kids=kids, qrow=qrow, loc=loc, spine_list=sp, spine_len=list(map(len, sp)),
                tok=draft_tokens, rr=idx[:n1], dd=idx[n1:2 * n1], qr=idx[2 * n1:3 * n1],
                spine=idx[3 * n1:].view(n1, width), logq=torch.cat(logq), ids=_vocab_map(layer, dev),
                vocab=layer.config.vocab_size)


class TreeView:
    """The last recorded tree as tables, float64: P[v] the target distribution after node v, Q[r] the
    draft distribution (over the target vocab) of the expanded node in row r = qrow[v]. One gather
    and one softmax each, no host sync."""

    def __init__(self, logits, logits_processor):
        x = logits[SIDE["rr"], SIDE["dd"]]
        if logits_processor is not None:
            x = logits_processor(None, x)
        self.P = torch.softmax(x.double(), dim=-1)
        lq = SIDE["logq"]
        self.Q = torch.zeros(lq.shape[0], SIDE["vocab"], dtype=torch.float64, device=lq.device)
        self.Q[:, SIDE["ids"]] = torch.softmax(lq.double(), dim=-1)
        for name in ("kids", "tokens", "qrow", "tok", "qr", "spine", "spine_len", "spine_list"):
            setattr(self, name, SIDE[name])
        self.ar = torch.arange(self.spine.shape[1], device=lq.device)
        self.zero = torch.zeros(1, dtype=torch.float64, device=lq.device)

    def p(self, u):
        """Target distribution of the token after node u."""
        return self.P[u]

    def q(self, u):
        """Draft distribution the children of node u were drawn from, over the target vocab."""
        return self.Q[self.qrow[u]]


def without(q, toks):
    """q with the tokens toks removed, renormalized. Zeroed one integer index at a time: a list index
    is copied to the GPU from pageable memory on every call."""
    if not toks:
        return q
    q = q.clone()
    for t in toks:
        q[t] = 0.0
    return q / q.sum()


def residual(target, q):
    """norm((target - q)_+); target itself if that is all zero (target == q). No host sync."""
    r = torch.clamp(target - q, min=0.0)
    s = r.sum()
    return torch.where(s > 0, r / s, target)


def rrsw(view, rand=random.random):
    """Token-level RRS without replacement. Returns the accepted node ids and the bonus distribution."""
    u, R, i, path = 0, view.p(0), 0, []
    while True:
        kids = view.kids[u]
        if i == len(kids):
            return path, R
        if i == 0:
            qi = view.q(u)
        t = view.tokens[kids[i]]
        Rt, qt = torch.stack((R[t], qi[t])).tolist()
        if qt <= 0.0:  # not a real draw: q had fewer than k tokens
            return path, R
        if rand() * qt < Rt:  # accept with min(1, R(t) / q_i(t))
            path.append(kids[i])
            u, R, i = kids[i], view.p(kids[i]), 0
        else:
            R, qi, i = residual(R, qi), without(qi, [t]), i + 1


def hsd(view, rand=random.random):
    """HSD Algorithm 3 on the tree. Returns the accepted node ids and the bonus distribution.
    Each path (child i, then first children down to a leaf) is block-verified with one batch of
    tensor ops and a single host sync."""
    u, R, i, used, path = 0, view.p(0), 0, [], []
    while True:
        kids = view.kids[u]
        if i == len(kids):
            return path, R
        c = kids[i]
        g, nodes, idx = view.spine_len[c], view.spine_list[c], view.spine[c, : view.spine_len[c]]
        P = torch.cat([R[None], view.P[idx]])  # P[m]: target at position m; P[g]: after the leaf
        Q = torch.cat([without(view.q(u), used)[None], view.Q[view.qr[idx[:-1]]]])  # Q[m]: draft at position m
        x, m = view.tok[idx], view.ar[:g]
        qx = Q[m, x]
        cs = torch.cat([view.zero, torch.cumsum(torch.log(P[m, x]) - torch.log(qx), 0)])
        rho_t = torch.exp(cs - torch.cummax(cs, 0).values)  # rho_m = min(rho_{m-1} p/q, 1), rho_0 = 1
        res = torch.clamp(rho_t[:g, None] * P[:g] - Q, min=0.0)  # residual numerators, row 0 = (R - q_i)_+
        vals = torch.cat([rho_t, res.sum(-1), qx[:1]]).tolist()  # the only host sync of this path
        rho, S, q0 = vals[: g + 1], vals[g + 1 : 2 * g + 1], vals[-1]
        if q0 <= 0.0:  # not a real draw
            return path, R
        if rand() < rho[g]:  # h_g = rho_g: the whole path is accepted
            return path + nodes, P[g]
        for j in range(g - 1, 0, -1):  # backward scan, h_j = S_j / (S_j + 1 - rho_j), 1 if rho_j = 1
            if rho[j] >= 1.0 or rand() * (S[j] + 1.0 - rho[j]) < S[j]:
                # stop at position j: keep nodes[:j] and go on at nodes[j - 1], whose first child is tried
                path += nodes[:j]
                u, R, i, used = nodes[j - 1], (res[j] / S[j] if S[j] > 0 else P[j]), 1, [view.tokens[nodes[j]]]
                break
        else:  # stop at position 0: try the next child of u
            R, i, used = (res[0] / S[0] if S[0] > 0 else P[0]), i + 1, used + [view.tokens[c]]


_PG, _BLOCK = 16, 4096  # longest path the fused walker takes; vocab entries per program

if triton is not None:
    @triton.jit(do_not_specialize=["g"])
    def _hsd_path_kernel(R, Q0, P, Q, nodes, qr, tok, out, g, V, sp, sq, BLOCK: tl.constexpr, G: tl.constexpr):
        """The tensor part of one hsd() path. Program (m, b) finds rho_m from the log ratios of positions
        0..m-1 and adds its vocab block's share of S_m = sum (rho_m P_m - Q_m)_+ to out[G + 1 + m]; block 0
        stores rho_m at out[m], program g - 1 also rho_g at out[g], program 0 also Q_0(x_0) at out[2G + 1].
        Position 0 uses R and Q0; position m >= 1 the target after nodes[m - 1] and its children's draft."""
        m = tl.program_id(0)
        b = tl.program_id(1)
        x0 = tl.load(tok + tl.load(nodes))
        q0x = tl.load(Q0 + x0)
        lr = tl.log(tl.load(R + x0)) - tl.log(q0x)  # log ratio of position 0 (-inf where R(x_0) = 0)
        cs = q0x * 0.0  # an fp64 zero (q0x is finite; lr * 0 is nan at -inf). cs_j: sum of the first j
        mx = cs  # log ratios; rho_j = exp(cs_j - max(cs_0..cs_j))
        rho_m = cs + 1.0
        last = tl.where(m == g - 1, g, m)
        for j in range(1, last + 1):
            if j > 1:  # log ratio of position j - 1
                u = tl.load(nodes + j - 2)
                x = tl.load(tok + tl.load(nodes + j - 1))
                lr = tl.log(tl.load(P + u * sp + x)) - tl.log(tl.load(Q + tl.load(qr + u) * sq + x))
            cs = cs + lr
            mx = tl.maximum(mx, cs)
            if j == m:
                rho_m = tl.exp(cs - mx)
            if j == g:
                if b == 0:
                    tl.store(out + g, tl.exp(cs - mx))
        if m == 0:
            prow = R
            qrow = Q0
        else:
            u = tl.load(nodes + m - 1)
            prow = P + u * sp
            qrow = Q + tl.load(qr + u) * sq
        offs = b * BLOCK + tl.arange(0, BLOCK)
        msk = offs < V
        p = tl.load(prow + offs, mask=msk, other=0.0)
        q = tl.load(qrow + offs, mask=msk, other=0.0)
        tl.atomic_add(out + G + 1 + m, tl.sum(tl.maximum(rho_m * p - q, 0.0), axis=0))
        if b == 0:
            tl.store(out + m, rho_m)
            if m == 0:
                tl.store(out + 2 * G + 1, q0x)


def _path_stats(view, R, q0, c, g):
    """rho_0..rho_g, S_0..S_{g-1} and q0(x_0) of the path that starts at child c: one launch, one sync."""
    assert g <= _PG, g
    out = torch.zeros(2 * _PG + 2, dtype=torch.float64, device=R.device)
    V = R.shape[0]
    _hsd_path_kernel[(g, triton.cdiv(V, _BLOCK))](R, q0, view.P, view.Q, view.spine[c], view.qr, view.tok, out, g, V,
                                                 view.P.stride(0), view.Q.stride(0), BLOCK=_BLOCK, G=_PG)
    vals = out.tolist()
    return vals[: g + 1], vals[_PG + 1 : _PG + 1 + g], vals[2 * _PG + 1]


def hsd_fused(view, rand=random.random):
    """hsd() with each path's ratios and residual masses from one _hsd_path_kernel launch instead of ~20
    small ops; the residual row is built only where the walk stops. Same rule and coins as hsd(), so
    the same walk up to float rounding (the residual sums are atomic adds)."""
    u, R, i, used, path = 0, view.p(0), 0, [], []
    while True:
        kids = view.kids[u]
        if i == len(kids):
            return path, R
        c = kids[i]
        g, nodes = view.spine_len[c], view.spine_list[c]
        q0 = without(view.q(u), used)
        rho, S, q0x = _path_stats(view, R, q0, c, g)
        if q0x <= 0.0:  # not a real draw
            return path, R
        if rand() < rho[g]:  # h_g = rho_g: the whole path is accepted
            return path + nodes, view.P[nodes[-1]]
        for j in range(g - 1, 0, -1):  # backward scan, h_j = S_j / (S_j + 1 - rho_j), 1 if rho_j = 1
            if rho[j] >= 1.0 or rand() * (S[j] + 1.0 - rho[j]) < S[j]:
                path += nodes[:j]
                w = nodes[j - 1]
                R = torch.clamp(rho[j] * view.P[w] - view.Q[view.qrow[w]], min=0.0) / S[j] if S[j] > 0 else view.P[w]
                u, i, used = w, 1, [view.tokens[nodes[j]]]
                break
        else:  # stop at position 0: try the next child of u
            R = torch.clamp(R - q0, min=0.0) / S[0] if S[0] > 0 else R
            i, used = i + 1, used + [view.tokens[c]]


def make_evaluate_posterior(walk, shadow=0, shadow_hsd=None):
    """evaluate_posterior for eagle.model.ea_model that verifies the last recorded tree with walk.
    Appends to STATS per step the length of the first-child path from the root and, with shadow > 0,
    each walker's mean accepted length over `shadow` extra walks on the same tree (own coins; the HSD
    side uses shadow_hsd, default hsd)."""

    def evaluate_posterior(logits, candidates, logits_processor, **_):
        assert logits_processor is not None, "sampled trees need temperature > 0"
        view = TreeView(logits, logits_processor)
        path, R = walk(view)
        u, spine = 0, 0
        while view.kids[u]:
            u, spine = view.kids[u][0], spine + 1
        rec = {"spine": spine}
        t0 = time.perf_counter()
        for name, w in (("rrsw", rrsw), ("hsd", shadow_hsd or hsd)) if shadow else ():
            rec[name] = sum(len(w(view, _SHADOW.random)[0]) for _ in range(shadow)) / shadow
        rec["shadow_s"] = time.perf_counter() - t0  # the walks end on a GPU sync; subtract from the timings
        STATS.append(rec)
        row, depth = SIDE["loc"][path[-1] if path else 0]
        return torch.tensor(row), depth, R

    return evaluate_posterior
