# HSD on sampled draft trees (EAGLE-3)

HSD only accepts more tokens than token-wise verification when the drafts are sampled, because only then do the
draft probabilities carry information a verifier can use. EAGLE-3 drafts its tree by top-k, so here the draft head
samples the tree instead, and three lossless verifiers are compared on it. See the main README for the results.

## Files

- `sampled_tree.py`: the tree builder and the verifiers.
  - `topK_genrate` replaces `eagle.model.cnets.Model.topK_genrate`. Each expanded node draws its k children from
    the draft distribution without replacement (Gumbel top-k). Nothing is pruned afterwards: dropping a node
    because of its own sampled score would bias the draws, while choosing which nodes to expand does not.
    `spines=m` keeps m of the k expansions per level on the paths of first-drawn children under the first m root
    children; `extra=e` lets the first such path go on for e more levels.
  - `rrsw`: token-level recursive rejection sampling without replacement.
  - `hsd`: tree HSD. It block-verifies the path from a child down its first-drawn children; when the walk stops at
    a node, it continues with that node's next child under the residual of the stop position.
  - `hsd_fused`: the same walk with each path's tensor work in one Triton kernel.
- `tree_bench.py`: runs EAGLE-3's own tree and the sampled trees back to back on one GPU, one JSON line per turn.
- `analyze_tree.py`: tokens per step, tokens per second and per-step timings, with paired bootstrap intervals over
  questions, and for runs with `--shadow` both verifiers on the same trees.
- `test_sampled_tree.py`: CPU checks. A: the walkers against the exact one-step laws (Monte Carlo). B: tree
  structure with the real EAGLE-3 head. C: the without-replacement draws. F: `hsd_fused` against `hsd`.
- `verify_hsd_exact.py`: exact enumeration of single-draft HSD (Eqs. 16-20) against block verification
  (Sun et al., 2024) and token-wise verification, including losslessness.
- `verify_sampled_tree_wor.py`, `verify_alg3_exact.py`: exact losslessness of the tree verifiers on sampled trees.

## Usage

Requirements: the EAGLE-3H requirements (transformers 4.47.1), PyTorch 2.x, and Triton for `hsd_fused`.

```bash
# exact checks, CPU
python verify_hsd_exact.py
python verify_sampled_tree_wor.py spines=1
python verify_sampled_tree_wor.py spines=1 extra=1

# CPU tests; part B needs local copies of the two models, part F runs Triton's interpreter
TRITON_INTERPRET=1 EAGLE3_HEAD=/path/to/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  BASE_MODEL=/path/to/Llama-3.1-8B-Instruct python test_sampled_tree.py

# benchmark: one shard per GPU, then the summary
mkdir -p runs/x
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python tree_bench.py --base /path/to/Llama-3.1-8B-Instruct \
    --ea /path/to/sglang-EAGLE3-LLaMA3.1-Instruct-8B --configs eagle,sampled_rrsw,hybrid1_rrsw,hybrid1_hsd \
    --shard $g --num-shards 8 --out runs/x/shard$g.jsonl &
done; wait
python analyze_tree.py runs/x
```

`tree_bench.py --help` and its docstring list the configurations; add `--shadow 8` to run both verifiers eight more
times per step on the same tree for the `*_hsd` configurations.
