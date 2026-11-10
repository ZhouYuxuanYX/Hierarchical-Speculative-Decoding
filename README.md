# Hierarchical Speculative Decoding (HSD)

**Hierarchical Speculative Decoding (HSD)** is a lossless verification algorithm for speculative decoding. It verifies a draft through the joint probability of its tokens and reconstructs the target model's distribution with a hierarchy of resampling distributions over branches, so the output follows the target model exactly.

For a single draft, HSD's lossless rule is the same rule as block verification ([Sun et al., 2024](https://arxiv.org/abs/2403.10444)): both accept every prefix of the draft with the same probability and resample from the same residual, so they accept the same number of tokens in expectation. The paper states this comparison as an inequality; it is an equality (see [Single draft: HSD and block verification coincide](#single-draft-hsd-and-block-verification-coincide)).

HSD reaches this rule from a different direction. It starts from the probability mass that each branch is missing or has in excess, and balances the two across branches. This gives an explicit residual distribution at every position of the draft, and the same residuals carry over to settings that block verification does not cover: several drafts verified in turn (Algorithm 3) and draft trees. On sampled draft trees for EAGLE-3, tree HSD accepts 0.5–3.3% more draft tokens per step than recursive rejection sampling on the same trees, and the best sampled-tree setup decodes 9.0% faster than EAGLE-3 with its default tree (see [EAGLE-3](#eagle-3)).

## What changed in this revision

This revision corresponds to arXiv v3 of the paper and brings the code and the README in line with the theory. The algorithms in the paper (Algorithms 2 and 3) were already lossless; the deviations were in the released implementation. Concretely:

- **Theorem 7 is strengthened.** The paper stated $`\mathbb{E}[\tau]_{\text{branch}} \geq \mathbb{E}[\tau]_{\text{block}} \geq \mathbb{E}[\tau]_{\text{token}}`$. The first inequality is an equality for every draft length $`\gamma`$; the previous remark that equality holds only for $`\gamma = 1`$ was removed. See the proof below and Appendix D.2/Appendix I of the paper.
- **Single-draft numbers are relabelled.** Tables 1 and 2 in this README were measured with the earlier implementation, whose full-draft test was not lossless. They are the **fast mode** numbers. Lossless single-draft HSD matches block verification, so its expected block efficiency is the Blockwise row.
- **Multi-draft numbers are not re-run.** Table 2's multi-draft results were measured with the earlier implementation (uncapped full-draft test plus the prefix carry-over). We have not re-run them with the corrected code; exact checks on small distributions are in [Multiple drafts](#multiple-drafts).
- **The EAGLE-3H numbers are gone.** EAGLE-3 drafts by top-k, so no lossless verifier can beat token-wise verification in expectation; the EAGLE-3H results in earlier README versions were measured before these fixes and were misleading. They are replaced by HSD on sampled EAGLE-3 draft trees ([EAGLE-3](#eagle-3)).

## Table of Contents

- [Single draft: HSD and block verification coincide](#single-draft-hsd-and-block-verification-coincide)
- [Lossless and Fast Modes](#lossless-and-fast-modes)
- [Single-draft results in the paper](#single-draft-results-in-the-paper)
- [Multiple drafts](#multiple-drafts)
- [EAGLE-3](#eagle-3)
- [Sampled Draft Trees for EAGLE-3](#sampled-draft-trees-for-eagle-3)
- [Reproduce the GSM8K experiments](#reproduce-the-gsm8k-experiments)
- [Citation](#citation)
- [References](#references)

## Single draft: HSD and block verification coincide

Write $`r_i`$ for the ratio $`p/q`$ of the $`i`$-th draft token and $`R_t = r_1 \cdots r_t`$, with $`R_0 = 1`$. Block verification keeps $`p_t = \min\{p_{t-1} r_t, 1\}`$, and by Lemma 11 of the paper, $`p_t = \min\{1, r_t, r_{t-1} r_t, \dots, r_1 \cdots r_t\} = R_t / \max_{0 \le s \le t} R_s`$. HSD's capped ratio $`r^*(\boldsymbol{X}_{1:t})`$ (Definition 5) divides $`R_t`$ by the running maximum over the shorter prefixes: it is $`\min\{r(\boldsymbol{X}_{1:m}), 1\} \cdot r(\boldsymbol{X}_{m+1:t})`$, where $`m`$ is the shorter prefix at which $`R`$ is largest. So $`\min\{r^*(\boldsymbol{X}_{1:t}), 1\} = p_t`$. Substituting this into Eq. 19 of the paper gives $`h_t = h_t^{\text{block}}`$ at every position, and the residual in Eq. 20 is the distribution block verification resamples from. In the comparison in Appendix D.2, the step $`\min\{r(x_{t+1}), r(\boldsymbol{X}_{t:t+1}), \dots, r(\boldsymbol{X}_{1:t+1})\} \le r(\boldsymbol{X}_{m+1:t+1})`$ holds with equality for the same reason, so no strict gain can follow from it.

`tree-hsd/verify_hsd_exact.py` checks this by enumerating every draft on small random target and draft distributions (7 configurations, 20 seeds each, draft lengths 2–4). The acceptance probabilities of Eq. 19 and of block verification differ by at most 1.5e-12 and the residuals by at most 6e-15; both reproduce the target distribution to within 4e-16 in total variation distance and give identical expected tokens per step.

## Lossless and Fast Modes

HSD runs in lossless mode by default. It accepts the full draft with probability $`\min\{r^*(\boldsymbol{X}_{1:\gamma}),1\}`$ (Eq. 19), where the capped ratio $`r^*`$ divides the joint ratio $`r(\boldsymbol{X}_{1:\gamma})`$ by its largest value over the shorter prefixes. The output then follows the target model's distribution exactly.

Fast mode accepts the full draft with the uncapped $`\min\{r(\boldsymbol{X}_{1:\gamma}),1\}`$ instead. It accepts the full draft at least as often, so it yields slightly more tokens per step, but its output deviates slightly from the target model's distribution. In exact checks on small synthetic distributions, fast mode produced 0.5–3% more tokens per step and shifted the output distribution by 0.06–0.15 in total variation distance. Enable it with `--fast` in `eval_speculative_decoding_llm.py`, or with `fast=True` in `generate`.

Earlier versions of this repository always used the fast rule, because the cap was missing from the full-draft test. Lossless is now the default; fast-mode runs save their results with `_fast` in the file name.

## Single-draft results in the paper

The HSD rows in the paper's Table 1 and Table 3 (a)/(b), and the figure of average acceptance probabilities, were measured with the earlier code, i.e. in fast mode. On GSM8K with Qwen2.5-0.5B as the draft model:

#### Table 1: Accuracy comparison (fast mode)

<table>
  <thead>
    <tr><th>Target Model</th><th>Tokenwise</th><th>HSD (ours)</th></tr>
  </thead>
  <tbody>
    <tr><td>Qwen2.5-72B</td><td>0.8213</td><td><strong>0.8517</strong></td></tr>
    <tr><td>Qwen2.5-32B</td><td>0.8213</td><td><strong>0.8479</strong></td></tr>
    <tr><td>Qwen2.5-14B</td><td>0.8327</td><td><strong>0.8327</strong></td></tr>
  </tbody>
</table>

The HSD column was measured with the earlier code, i.e. in fast mode, which does not preserve the target distribution exactly, so the differences between the columns should not be read as an accuracy gain.

#### Table 2: Block efficiency and decoding speed (fast mode)

| Method | BE 14B | BE 32B | BE 72B | DS 14B | DS 32B | DS 72B |
|---|---|---|---|---|---|---|
| Tokenwise | 5.99 | 6.14 | 6.44 | 82.28 | 53.87 | 31.49 |
| Blockwise | 6.13 (+2.3%) | 6.26 (+2.0%) | 6.53 (+1.4%) | 86.06 (+4.6%) | 54.91 (+1.9%) | 31.79 (+1.0%) |
| **HSD (Ours)** | **6.30 (+5.2%)** | **6.47 (+5.4%)** | **6.65 (+3.3%)** | **91.05 (+10.7%)** | **57.12 (+6.0%)** | **32.52 (+3.3%)** |

The HSD row was measured with the earlier code, i.e. in fast mode, where it improved BE by 3.3%–5.4% and DS by up to 10.7% over Tokenwise. In lossless mode, single-draft HSD is the same rule as block verification, so its expected BE is that of the Blockwise row: **1.4%–2.3% above Tokenwise** (on average 3.1% block efficiency and 3.0% decoding speed across the paper's Table 1). The margins over Blockwise in the fast-mode rows come from the uncapped test and from run-to-run variation.

## Multiple drafts

Algorithm 3 extends HSD to several independently sampled drafts: each later draft restarts the joint ratio at the accepted prefix, and the drafts are verified in turn. Earlier versions carried the previous draft's prefix ratio over, which biased the output slightly; the released code now follows Algorithm 3. The multi-draft results in the paper (Table 2) were measured with the earlier implementation, and we have not re-run them with the corrected code.

`tree-hsd/verify_alg3_exact.py` checks the corrected rule exactly on small synthetic distributions (2 or 3 independent drafts of length 2 or 3): the corrected Algorithm 3 reproduces the target distribution to within 5e-16 in total variation distance and accepts 0.3–3.3% more tokens per step than recursive rejection sampling with replacement. The earlier implementation accepted 1.1–3.2% more tokens per step than the corrected one and deviated from the target distribution by 0.09–0.26 in total variation distance, almost entirely because of the uncapped full-draft test; the carried-over ratio alone accounts for at most 0.02.

## EAGLE-3

EAGLE-3 fine-tunes an additional component to predict draft tokens from the target model's hidden features, eliminating the need for a separate draft model. HSD operates orthogonally to the drafting phase, so it works with a pre-fine-tuned EAGLE-3 model **without any additional fine-tuning**.

What HSD needs from the drafter is sampled drafts. EAGLE-3 drafts its tree by top-k, so every drafted token has draft probability 1, and with such deterministic drafts, at any sampling temperature, no lossless verifier accepts a drafted prefix more often than the target model would generate it: token-wise verification already reaches that bound. We therefore let EAGLE-3's draft head sample its tree and compare tree HSD with recursive rejection sampling without replacement (RRSw), the usual lossless verifier for sampled draft trees, on the same trees. Tree HSD accepts more tokens than RRSw on every tree shape we tried, and with the fused verifier it is the fastest setup we tried: on Llama-3.1-8B-Instruct it decodes 2.9% faster than RRSw on the same tree shape and 9.0% faster than EAGLE-3 with its default tree. The setup and all numbers are in [Sampled Draft Trees for EAGLE-3](#sampled-draft-trees-for-eagle-3), and the code is in [`tree-hsd/`](tree-hsd/README.md).

`EAGLE-3H/` keeps the integration with EAGLE-3's top-k trees, where HSD replaces the token-wise verifier:

```bash
cd EAGLE-3H
bash eagle_eval.sh
```

With the default top-k trees, EAGLE-3H accepts the same number of tokens per step as EAGLE-3 in expectation; the speed difference, if any, comes from the implementation, not from the acceptance rule.

## Sampled Draft Trees for EAGLE-3

HSD only adds accepted tokens when the drafts are sampled from the draft distribution, since only then do the ratios $`p/q`$ carry information a verifier can use. `tree-hsd/` makes EAGLE-3's draft head sample its tree: every expanded node draws its children from the draft distribution without replacement, and no node is pruned afterwards. Two lossless verifiers run on these trees:

- **RRSw**: token-level recursive rejection sampling without replacement, the multi-draft form of token-wise verification.
- **Tree HSD**: block-verify the path from a child down its first-drawn children with HSD's rule; if the walk stops at a node, continue with that node's next child under the residual of the stop position.

Both are exact: `tree-hsd/verify_sampled_tree_wor.py` reproduces the target distribution by enumeration on the four tree shapes used below (sampled, with a spine, and with the spine extended by 1 or 2 levels); the maximum total-variation distance to the target is 4.7e-15.

### Setup

Target Llama-3.1-8B-Instruct with the EAGLE-3 head `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`; target and draft at temperature 1; the six tasks used by EAGLE (MT-bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail summarization, Natural Questions), the first 80 questions of each, two seeds; batch size 1, all configurations back to back on the same GPU. Block efficiency (tokens per step) does not depend on the hardware; decoding speed is reported relative to EAGLE-3 on the same GPU. Intervals are 95% bootstrap intervals over questions.

Trees: 4 children per expanded node, 4 expanded nodes per level, 6 draft levels, 84 nodes, nothing pruned (draft length 7 with the root token, as in EAGLE-3H).
- *sampled*: the 4 nodes with the highest path log-probability are expanded, as EAGLE-3 does.
- *hybrid*: one of the 4 expansions per level continues the path of first-drawn children under the first root child (a *spine*), so that path always reaches the last level; the other three go by path log-probability.

EAGLE-3's default tree has 10 children per node and depth 5 and is pruned to 59 nodes; it uses token-wise verification.

### Table 3: Sampled trees vs EAGLE-3's default tree

| Tree | Verifier | Block Eff. | Decoding Speed vs EAGLE-3 |
|---|---|---|---|
| EAGLE-3 default | token-wise | 4.22 | — |
| sampled | RRSw | 4.37 (+3.6%) | +6.0% |
| hybrid | RRSw | 4.38 (+3.8%) | +6.0% |
| hybrid | tree HSD | 4.51 (+6.9%) | **+9.0%** |
| hybrid, spine 2 levels longer | RRSw | 4.67 (+10.7%) | +4.9% |
| hybrid, spine 2 levels longer | tree HSD | 4.77 (+13.1%) | +7.1% |
| hybrid, spine 4 levels longer | RRSw | 4.67 (+10.6%) | −2.4% |
| hybrid, spine 4 levels longer | tree HSD | **4.93 (+16.8%)** | +3.0% |

Tree HSD runs with one Triton kernel per verified path (`hsd_fused`); its verification takes 0.77 ms per step, against 0.81 ms for RRSw and 1.68 ms for EAGLE-3's verification. On the hybrid tree it is ahead of EAGLE-3 by 6.9% (95% CI +3.5% to +10.4%) in block efficiency and by 9.0% (+5.6% to +12.5%) in decoding speed. Against RRSw on the same tree shape it adds 2.9% in block efficiency (+1.1% to +4.7%) and 2.9% in decoding speed (+1.2% to +4.7%). The rest of the gain over EAGLE-3 comes from sampling the tree and from cheaper verification: sampled RRSw is already 6.0% faster than EAGLE-3, of which about 2 points come from the tensorized verifier (0.81 ms per step instead of 1.68 ms) and the rest from the sampled tree shape.

The spine is the path tree HSD verifies first, and making it longer widens HSD's lead. With 4 more levels, RRSw gains nothing over 2 more levels (4.67 tokens per step either way), while tree HSD goes from 4.77 to 4.93 and leads RRSw by 5.6% (+3.5% to +7.8%) in block efficiency and 5.5% (+3.4% to +7.7%) in speed. Each extra level costs one more forward pass of the draft head, about 1.5 ms per step in this implementation, so the longer spines are not faster than the plain hybrid tree here. With a cheaper draft pass the longer spine could come out ahead; we have not measured that.

### Table 4: Both verifiers on the same trees

Different configurations generate different text, which adds noise to the comparisons above. To compare the two verifiers directly, every step also ran both of them 8 more times on the same tree with separate random numbers. Accepted draft tokens per step, tree HSD vs RRSw:

| Tree | Tree HSD vs RRSw |
|---|---|
| sampled | +0.5% (+0.3% to +0.6%) |
| hybrid | +1.5% (+1.3% to +1.7%) |
| hybrid with two spines | +1.7% (+1.5% to +1.8%) |
| all expansions on first-child paths | +3.0% (+2.8% to +3.2%) |
| hybrid, spine 2 levels longer | +2.6% (+2.4% to +2.9%) |
| hybrid, spine 4 levels longer | +3.3% (+3.1% to +3.6%) |

HSD's joint test acts along first-child paths, so the gain grows with their length; in the sampled tree, the path under the first root child reaches the last level in only 43% of the steps. The gain also grows with the entropy of the task: on the hybrid tree it goes from +0.5% on HumanEval to +3.0% on Natural Questions, and with the spine 4 levels longer from +2.2% to +5.3%. Trees whose expansions all continue first-child paths favor HSD but are weaker trees overall (RRSw loses 5.1% against the sampled tree). The end-to-end gaps in Table 3 are larger than these same-tree gaps. The same-tree numbers isolate the verifier; the end-to-end numbers also depend on where each verifier's steps end, which changes the trees drafted next.

### Table 5: Single chain in SGLang

SGLang implements block verification (`--speculative-use-block-verification`), which is single-draft HSD in lossless mode. EAGLE-3 chain drafts sampled at temperature 1, batch size 1, first turns of the same tasks:

| Draft length | Block Eff., token-wise | Block Eff., block (= HSD) | Gain | Speed, token-wise | Speed, block (= HSD) |
|---|---|---|---|---|---|
| 5 | 3.26 | 3.39 | +4.0% (+2.2% to +5.9%) | 1.00 | 1.03 |
| 10 | 3.65 | 3.83 | +4.8% (+3.1% to +6.5%) | 0.91 | 0.95 |
| 15 | 3.71 | 3.92 | +5.6% (+3.3% to +7.9%) | 0.79 | 0.83 |

Speed is relative to token-wise verification at draft length 5. The gain of block verification grows with the draft length, but longer chains are slower overall with this draft head, so the fastest setting is draft length 5 with block verification.

Tables 3 and 4 can be reproduced with [`tree-hsd/`](tree-hsd/README.md). Table 5 uses SGLang's server flags `--speculative-algorithm EAGLE3 --speculative-eagle-topk 1 --speculative-num-steps N --speculative-num-draft-tokens N+1 --speculative-use-rejection-sampling`, plus `--speculative-use-block-verification` for block verification; sampling the drafts of an EAGLE-3 head with a reduced draft vocabulary needs [sgl-project/sglang#40893](https://github.com/sgl-project/sglang/pull/40893).

## Reproduce the GSM8K experiments

### Requirements

Install `python=3.10` and `transformers=4.46.3`, then copy the provided files under the `transformers` directory into your `transformers` installation, e.g.:

```
anaconda3/envs/your_environment_name/lib/python3.10/site-packages/transformers/
```

Then navigate to `chain-of-thought-hub/gsm8k`:

```bash
cd chain-of-thought-hub/gsm8k
```

### Single-draft verification

Baseline:

```bash
bash eval_speculative_qwen.sh
```

HSD (lossless):

```bash
bash eval_speculative_qwen_backward_clever.sh
```

HSD in fast mode (uncapped full-draft test, not lossless):

```bash
python3 eval_speculative_decoding_llm.py --backward --speculative --clever --fast
```

### Multi-draft verification

Each later draft restarts the joint ratio at the accepted prefix, as in Algorithm 3.

```bash
bash eval_speculative_qwen_multidraft_11.sh        # tokenwise multidraft
bash eval_speculative_qwen_backward_clever_multidraft_11.sh   # HSD multidraft
python3 eval_speculative_decoding_llm.py --backward --speculative --clever --multidraft 11 --parallel --fast   # fast mode
```

### Compute metrics

```bash
python compute_speculative_stats.py
```

## Citation

```bibtex
@misc{zhou2026overcomingjointintractabilitylossless,
      title={Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding}, 
      author={Yuxuan Zhou and Fei Huang and Heng Li and Fengyi Wu and Tianyu Wang and Jianwei Zhang and Junyang Lin and Zhi-Qi Cheng},
      year={2026},
      eprint={2601.05724},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.05724}, 
}
```

## References

1. [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)  
2. [EAGLE-3 Repository](https://github.com/SforAiD/LLM-Eagle)
3. [Block Verification Accelerates Speculative Decoding (Sun et al., 2024)](https://arxiv.org/abs/2403.10444)
4. [SGLang](https://github.com/sgl-project/sglang)
