# Hierarchical Speculative Decoding (HSD)

**Hierarchical Speculative Decoding (HSD)** is a lossless verification algorithm for speculative decoding. It verifies a draft through the joint probability of its tokens and reconstructs the target model's distribution with a hierarchy of resampling distributions over branches, so the output follows the target model exactly.

For a single draft, HSD's lossless rule is the same rule as block verification ([Sun et al., 2024](https://arxiv.org/abs/2403.10444)): both accept every prefix of the draft with the same probability and resample from the same residual, so they accept the same number of tokens in expectation. The paper states this comparison as an inequality; it is an equality (see [Corrections](#corrections)).

HSD reaches this rule from a different direction. It starts from the probability mass that each branch is missing or has in excess, and balances the two across branches. This gives an explicit residual distribution at every position of the draft, and the same residuals carry over to settings that block verification does not cover: several drafts verified in turn (Algorithm 3) and draft trees. On sampled draft trees for EAGLE-3, tree HSD accepts 0.5–3.3% more draft tokens per step than recursive rejection sampling on the same trees, and the best sampled-tree setup decodes 9.0% faster than EAGLE-3 with its default tree (see [Sampled Draft Trees for EAGLE-3](#sampled-draft-trees-for-eagle-3)).

## Table of Contents

- [Corrections](#corrections)
- [Experiment Settings](#experiment-settings)
- [Lossless and Fast Modes](#lossless-and-fast-modes)
- [Reproduce Experiment Results on GSM8K](#reproduce-experiment-results-on-gsm8k)
- [EAGLE Integration](#eagle-integration)
- [Sampled Draft Trees for EAGLE-3](#sampled-draft-trees-for-eagle-3)
- [Citation](#citation)
- [References](#references)

## Corrections

### Single draft: HSD and block verification coincide

Theorem 7 of the paper states $`\mathbb{E}[\tau]_{\text{branch}} \geq \mathbb{E}[\tau]_{\text{block}} \geq \mathbb{E}[\tau]_{\text{token}}`$. The first inequality is an equality for every draft length $`\gamma`$.

Write $`r_i`$ for the ratio $`p/q`$ of the $`i`$-th draft token and $`R_t = r_1 \cdots r_t`$, with $`R_0 = 1`$. Block verification keeps $`p_t = \min\{p_{t-1} r_t, 1\}`$, and by Lemma 11, $`p_t = \min\{1, r_t, r_{t-1} r_t, \dots, r_1 \cdots r_t\} = R_t / \max_{0 \le s \le t} R_s`$. HSD's capped ratio $`r^*(\boldsymbol{X}_{1:t})`$ (Definition 5) divides $`R_t`$ by the running maximum over the shorter prefixes: it is $`\min\{r(\boldsymbol{X}_{1:m}), 1\} \cdot r(\boldsymbol{X}_{m+1:t})`$, where $`m`$ is the shorter prefix at which $`R`$ is largest. So $`\min\{r^*(\boldsymbol{X}_{1:t}), 1\} = p_t`$. Substituting this into Eq. 19 gives $`h_t = h_t^{\text{block}}`$ at every position, and the residual in Eq. 20 is the distribution block verification resamples from. In the comparison in Appendix D.2, the step $`\min\{r(x_{t+1}), r(\boldsymbol{X}_{t:t+1}), \dots, r(\boldsymbol{X}_{1:t+1})\} \le r(\boldsymbol{X}_{m+1:t+1})`$ holds with equality for the same reason, so no strict gain can follow from it.

`tree-hsd/verify_hsd_exact.py` checks this by enumerating every draft on small random target and draft distributions (7 configurations, 20 seeds each, draft lengths 2–4). The acceptance probabilities of Eq. 19 and of block verification differ by at most 1.5e-12 and the residuals by at most 6e-15; both reproduce the target distribution to within 4e-16 in total variation distance and give identical expected tokens per step.

### Single-draft results in Tables 1 and 2

These were measured with the earlier code, which tested the full draft with the uncapped ratio. That rule is now fast mode and is not lossless; see [Lossless and Fast Modes](#lossless-and-fast-modes). In lossless mode, single-draft HSD accepts as many tokens as block verification, i.e. the Blockwise row of Table 2.

### Multiple drafts

Each later draft now restarts the joint ratio at the accepted prefix, as in Algorithm 3. Earlier versions carried the previous draft's prefix ratio over, which biased the output slightly.

### EAGLE-3H

EAGLE-3 drafts the top-k tokens, so every drafted token has draft probability 1. With such deterministic drafts, at any sampling temperature, no lossless verifier can accept a drafted prefix more often than the target model would generate it, and token-wise verification already reaches that bound, so HSD cannot add accepted tokens. Earlier versions of EAGLE-3H also carried the accepted prefix's joint probability into the next tree path, which biased the output. The EAGLE-3H results in earlier versions of this README were measured before these fixes and have been removed. HSD improves EAGLE-3 once the draft tree is sampled; see [EAGLE Integration](#eagle-integration).

## Experiment Settings
### Experiments Setup
Experiments are conducted with the widely adopted GPTQ-quantized 8-bit instruction-tuned Qwen2.5 series [Bai et al., 2023](#). By default, we employ the 0.5B model as the draft model and 72B as the target model, with a temperature of 1. All experiments were conducted on a single NVIDIA H20 GPU with 96 GB of memory, unless otherwise specified.

### Baselines and Metrics
We compare two lossless verification methods—Token-wise [Leviathan et al., 2023](https://arxiv.org/abs/2211.17192) and Block-wise [Sun et al., 2024](https://arxiv.org/abs/2403.10444)—using two metrics: *Block Efficiency* (tokens/step) and *Decoding Speed* (tokens/second).
- **Block Efficiency** measures the average tokens generated per serial call to the target model, reflecting intrinsic efficiency independent of hardware.  
- **Decoding Speed** indicates tokens produced per second for practical reference, though it depends on hardware and implementation.

## Lossless and Fast Modes

HSD runs in lossless mode by default. It accepts the full draft with probability $`\min\{r^*(\boldsymbol{X}_{1:\gamma}),1\}`$ (Eq. 19), where the capped ratio $`r^*`$ divides the joint ratio $`r(\boldsymbol{X}_{1:\gamma})`$ by its largest value over the shorter prefixes. The output then follows the target model's distribution exactly.

Fast mode accepts the full draft with the uncapped $`\min\{r(\boldsymbol{X}_{1:\gamma}),1\}`$ instead. It accepts the full draft at least as often, so it yields slightly more tokens per step, but its output deviates slightly from the target model's distribution. In exact checks on small synthetic distributions, fast mode produced 0.5–3% more tokens per step and shifted the output distribution by 0.06–0.15 in total variation distance. Enable it with `--fast` in `eval_speculative_decoding_llm.py`, or with `fast=True` in `generate`.

Earlier versions of this repository always used the fast rule, because the cap was missing from the full-draft test. Tables 1 and 2 below were measured with that earlier code, so they correspond to fast mode. After updating, copy `transformers/generation/utils.py` into your `transformers` installation again.

## Reproduce Experiment Results on GSM8K

### Requirements
Please install `python=3.10` and `transformers=4.46.3`, then copy the provided files under the `transformers` directory to:

```
anaconda3/envs/your_environment_name/lib/python3.10/site-packages/transformers/
```
Then navigate to `chain-of-thought-hub/gsm8k`:
```bash
cd chain-of-thought-hub/gsm8k
```

### Single-Draft Verification

Baseline methods:
- **Tokenwise** verification
  ```bash
  bash eval_speculative_qwen.sh
  ```

Our methods:

- **HSD (clever)** - optimized with smart capping mechanism
  ```bash
  bash eval_speculative_qwen_backward_clever.sh
  ```

- **HSD (clever), fast mode** - uncapped full-draft test, not lossless (see [Lossless and Fast Modes](#lossless-and-fast-modes))
  ```bash
  python3 eval_speculative_decoding_llm.py --backward --speculative --clever --fast
  ```

### Multi-Draft Verification with Recursive Reject Sampling

Extends HSD to use multiple independently sampled drafts for even better performance. Each later draft restarts the joint ratio at the accepted prefix, as in Algorithm 3. Earlier versions carried the previous draft's prefix ratio over, which also biased the output slightly.

- **Tokenwise** multidraft
  ```bash
  bash eval_speculative_qwen_multidraft_11.sh
  ```

- **HSD** multidraft
  ```bash
  bash eval_speculative_qwen_backward_clever_multidraft_11.sh
  ```

- **HSD** multidraft, fast mode
  ```bash
  python3 eval_speculative_decoding_llm.py --backward --speculative --clever --multidraft 11 --parallel --fast
  ```

Fast-mode runs save their results with `_fast` in the file name.

### Compute Metrics

Finally, evaluate Block Efficiency and Decoding Speed:
```bash
python compute_speculative_stats.py
```

### GSM8K Performance Results

Using Qwen2.5-0.5B as draft model on GSM8K:

#### **Table 1**: Accuracy Comparison

<table>
  <thead>
    <tr>
      <th>Target Model</th>
      <th>Tokenwise</th>
      <th>HSD (ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-72B</td>
      <td>0.8213</td>
      <td><strong>0.8517</strong></td>
    </tr>
    <tr>
      <td>Qwen2.5-32B</td>
      <td>0.8213</td>
      <td><strong>0.8479</strong></td>
    </tr>
    <tr>
      <td>Qwen2.5-14B</td>
      <td>0.8327</td>
      <td><strong>0.8327</strong></td>
    </tr>
  </tbody>
</table>

The HSD column was measured with the earlier code, i.e. in fast mode, which does not preserve the target distribution exactly, so the differences between the columns should not be read as an accuracy gain.
#### **Table 2**: Block Efficiency & Speed 
| Method         | BE 14B | BE 32B | BE 72B | DS 14B | DS 32B | DS 72B |
|----------------|--------|--------|--------|--------|--------|--------|
| Tokenwise      | 5.99   | 6.14   | 6.44   | 82.28  | 53.87  | 31.49  |
| Blockwise      | 6.13 (+2.3%) | 6.26 (+2.0%) | 6.53 (+1.4%) | 86.06 (+4.6%) | 54.91 (+1.9%) | 31.79 (+1.0%) |
| **HSD (Ours)** | **6.30 (+5.2%)** | **6.47 (+5.4%)** | **6.65 (+3.3%)** | **91.05 (+10.7%)** | **57.12 (+6.0%)** | **32.52 (+3.3%)** |


The HSD row was measured with the earlier code, i.e. in fast mode, where it improved BE by 3.3%–5.4% and DS by up to 10.7% over Tokenwise. Lossless single-draft HSD is the same rule as block verification (see [Corrections](#corrections)), so its expected BE is that of the Blockwise row: 1.4%–2.3% above Tokenwise.

## EAGLE Integration

HSD can be integrated with [EAGLE-3](https://github.com/SforAiD/LLM-Eagle). EAGLE-3 fine-tunes an additional component to predict draft tokens from the target model's hidden features, eliminating the need for a separate draft model and thereby improving decoding speed. HSD operates orthogonally to the drafting phase, so it works with a pre-fine-tuned EAGLE-3 model **without any additional fine-tuning**.

What HSD needs from the drafter is sampled drafts. EAGLE-3 drafts its tree by top-k, so every drafted token has draft probability 1, and with such deterministic drafts no lossless verifier accepts more tokens than token-wise verification (see [Corrections](#corrections)). We therefore let EAGLE-3's draft head sample its tree and compare tree HSD with recursive rejection sampling without replacement (RRSw), the usual lossless verifier for sampled draft trees, on the same trees. Tree HSD accepts more tokens than RRSw on every tree shape we tried, and with the fused verifier it is the fastest setup we tried: on Llama-3.1-8B-Instruct it decodes 2.9% faster than RRSw on the same tree shape and 9.0% faster than EAGLE-3 with its default tree. The setup and all numbers are in [Sampled Draft Trees for EAGLE-3](#sampled-draft-trees-for-eagle-3), and the code is in [`tree-hsd/`](tree-hsd/README.md).

`EAGLE-3H/` keeps the integration with EAGLE-3's top-k trees, where HSD replaces the token-wise verifier:

```bash
cd EAGLE-3H
bash eagle_eval.sh
```

## Sampled Draft Trees for EAGLE-3

HSD only adds accepted tokens when the drafts are sampled from the draft distribution, since only then do the ratios $`p/q`$ carry information a verifier can use. `tree-hsd/` makes EAGLE-3's draft head sample its tree: every expanded node draws its children from the draft distribution without replacement, and no node is pruned afterwards. Two lossless verifiers run on these trees:

- **RRSw**: token-level recursive rejection sampling without replacement, the multi-draft form of token-wise verification.
- **Tree HSD**: block-verify the path from a child down its first-drawn children with HSD's rule; if the walk stops at a node, continue with that node's next child under the residual of the stop position.

Both are exact: `tree-hsd/verify_sampled_tree_wor.py` reproduces the target distribution by enumeration to within 3e-15 in total variation distance.

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

Tree HSD runs with one Triton kernel per verified path (`hsd_fused`); its verification takes 0.77 ms per step, against 0.81 ms for RRSw and 1.68 ms for EAGLE-3's verification. On the hybrid tree it is ahead of EAGLE-3 by 6.9% (95% CI +3.5% to +10.4%) in block efficiency and by 9.0% (+5.6% to +12.5%) in decoding speed. Against RRSw on the same tree shape it adds 2.9% in block efficiency (+1.1% to +4.7%) and 2.9% in decoding speed (+1.2% to +4.7%); the rest of the gain over EAGLE-3 comes from sampling the tree and from cheaper verification.

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

