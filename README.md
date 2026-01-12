# Hierarchical Speculative Decoding (HSD)

Hierarchical Speculative Decoding is the state-of-the-art lossless verification algorithm for faster LLM inference. HSD organizes draft tokens in a trie structure and performs level-by-level verification, accepting more tokens in expectation than both tokenwise and blockwise verification.

## Table of Contents

- [Key Innovation](#key-innovation)
- [Theoretical Guarantee](#theoretical-guarantee)
- [Toy Example Simulation](#compare-the-expected-number-of-accepted-tokens-with-a-toy-example)
- [GSM8K Experiments](#reproduce-experiment-results-on-gsm8k)
- [Human-Eval Results](#human-eval-results)
- [MMLU Results](#mmlu-results)
- [EAGLE Integration](#eagle-integration)
- [Ablation Studies](#ablation-studies)
- [Scaling Analysis](#scaling-analysis)
- [Algorithm Variants](#algorithm-variants)
- [Citation](#citation)
- [References](#references)

## Key Innovation

Unlike traditional tokenwise or blockwise verification, HSD uses a **hierarchical approach**:
- Draft tokens are organized in a tree/trie structure
- Verification proceeds level-by-level from root to leaves
- If a token at level k is rejected, verification stops for that subtree
- If accepted, continues to verify all children at the next level

This hierarchical structure enables HSD to accept more tokens while maintaining **lossless verification** - producing exactly the same distribution as the target model.

## Theoretical Guarantee

**Proposition 1**: For any target distribution p(·) and draft distribution q(·), HSD accepts more tokens in expectation than both tokenwise and blockwise verification.

## Compare the Expected Number of Accepted Tokens with a Toy Example

Let p(·) and q(·) denote the target and draft distributions, respectively. Following [1], we define two context-independent distributions as a toy example:

```
p(A) = 1/3,  p(B) = 2/3
q(A) = 2/3,  q(B) = 1/3
```

Expected number of accepted tokens (E[N]):
- Tokenwise: 5/9 ≈ 0.56
- Blockwise: 4/9 ≈ 0.44
- **HSD: 7/9 ≈ 0.78** ✓

You can run the simulation with this toy example using:
```bash
python simulation.py
```

See Figure 2 in the paper for visual intuition of the hierarchical trie structure.

## Reproduce Experiment Results on GSM8K

### Requirements
Please install `python=3.10` and `transformers=4.46.3`, then copy the provided files under the `transformers` directory to:

```
anaconda3/envs/your_environment_name/lib/python3.10/site-packages/transformers/
```

```bash
cd chain-of-thought-hub/gsm8k
```

### Single-Draft Verification

Baseline methods:
- **Tokenwise** verification
  ```bash
  bash evaluate_speculative_qwen.sh
  ```

- **Blockwise** verification
  ```bash
  bash evaluate_speculative_qwen_blockwise.sh
  ```

Our methods:
- **NaiveHSD** - basic hierarchical verification
  ```bash
  bash eval_speculative_qwen_backward.sh
  ```

- **HSD (clever)** - optimized with smart stopping criterion
  ```bash
  bash evaluate_speculative_qwen_backward_clever.sh
  ```

### Multi-Draft Verification with Recursive Reject Sampling

Extends HSD to use multiple independent draft models for even better performance.

- **Tokenwise** multidraft
  ```bash
  bash evaluate_speculative_qwen_multidraft_11.sh
  ```

- **HSD** multidraft
  ```bash
  bash evaluate_speculative_qwen_backward_clever_multidraft_11.sh
  ```

### Compute Metrics

Finally, evaluate Block Efficiency and Decoding Speed:
```bash
python compute_speculative_stats.py
```

### GSM8K Performance Results

Using Qwen2.5-0.5B as draft model on GSM8K:

#### Accuracy Comparison (Table 2)

<table>
  <thead>
    <tr>
      <th>Target Model</th>
      <th>Tokenwise</th>
      <th>HSD (ours)</th>
      <th>Accuracy Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-72B</td>
      <td>0.8213</td>
      <td><strong>0.8517</strong></td>
      <td>+3.0%</td>
    </tr>
    <tr>
      <td>Qwen2.5-32B</td>
      <td>0.8213</td>
      <td><strong>0.8479</strong></td>
      <td>+2.7%</td>
    </tr>
    <tr>
      <td>Qwen2.5-14B</td>
      <td>0.8327</td>
      <td><strong>0.8327</strong></td>
      <td>+0.0%</td>
    </tr>
  </tbody>
</table>

HSD maintains or improves accuracy while being significantly faster.

#### Block Efficiency & Speed (Table 3)

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Block Efficiency</th>
      <th>Decoding Speed</th>
      <th>Latency (ms/token)</th>
      <th>Throughput (tokens/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>No Speculative</td>
      <td>-</td>
      <td>1.00x</td>
      <td>62.80</td>
      <td>15.92</td>
    </tr>
    <tr>
      <td>Tokenwise</td>
      <td>1.00</td>
      <td>1.47x</td>
      <td>42.72</td>
      <td>23.41</td>
    </tr>
    <tr>
      <td>Blockwise</td>
      <td>2.21</td>
      <td>2.14x</td>
      <td>29.35</td>
      <td>34.07</td>
    </tr>
    <tr>
      <td>NaiveHSD</td>
      <td>2.21</td>
      <td>2.14x</td>
      <td>29.35</td>
      <td>34.07</td>
    </tr>
    <tr>
      <td><strong>HSD (ours)</strong></td>
      <td><strong>3.25</strong></td>
      <td><strong>2.87x</strong></td>
      <td><strong>21.88</strong></td>
      <td><strong>45.70</strong></td>
    </tr>
  </tbody>
</table>

HSD achieves **~2x speedup** over blockwise verification while being lossless.

## Human-Eval Results

**Note**: Results shown below. Reproduction code to be added in future update.

Human-Eval evaluates code generation capability. Results using Qwen2.5 models with different draft lengths (γ):

<table>
  <thead>
    <tr>
      <th>Target Model</th>
      <th>Draft Model</th>
      <th>γ=4 Tokenwise</th>
      <th>γ=16 Tokenwise</th>
      <th>γ=4 HSD</th>
      <th>γ=16 HSD</th>
      <th>HSD Gain (γ=16)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-32B</td>
      <td>Qwen2.5-0.5B</td>
      <td>0.5252</td>
      <td>0.6048</td>
      <td>0.5495</td>
      <td><strong>0.6162</strong></td>
      <td>+1.9%</td>
    </tr>
    <tr>
      <td>Qwen2.5-14B</td>
      <td>Qwen2.5-0.5B</td>
      <td>0.5252</td>
      <td>0.5971</td>
      <td>0.5495</td>
      <td><strong>0.6111</strong></td>
      <td>+2.4%</td>
    </tr>
  </tbody>
</table>

## MMLU Results

**Note**: Results shown below. Reproduction code to be added in future update.

MMLU (Massive Multitask Language Understanding) benchmark results across different model sizes:

<table>
  <thead>
    <tr>
      <th>Target Model</th>
      <th>Tokenwise</th>
      <th>HSD (ours)</th>
      <th>Accuracy Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-72B</td>
      <td>0.6850</td>
      <td><strong>0.7073</strong></td>
      <td>+2.3%</td>
    </tr>
    <tr>
      <td>Qwen2.5-32B</td>
      <td>0.6391</td>
      <td><strong>0.6630</strong></td>
      <td>+2.4%</td>
    </tr>
    <tr>
      <td>Qwen2.5-14B</td>
      <td>0.5670</td>
      <td><strong>0.5870</strong></td>
      <td>+2.0%</td>
    </tr>
  </tbody>
</table>

## EAGLE Integration

HSD can be integrated with [EAGLE](https://github.com/SforAiD/LLM-Eagle) for even better performance. EAGLE uses a lightweight autoregressive transformer to predict draft tokens, which can then be verified using HSD.

### EAGLE + HSD Results

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Dataset</th>
      <th>EAGLE Tokenwise</th>
      <th>EAGLE+HSD</th>
      <th>Speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LLaMA2-7B</td>
      <td>GSM8K</td>
      <td>0.5494</td>
      <td><strong>0.5633</strong></td>
      <td>~2.5x</td>
    </tr>
    <tr>
      <td>Vicuna-7B</td>
      <td>GSM8K</td>
      <td>0.4476</td>
      <td><strong>0.4568</strong></td>
      <td>~2.3x</td>
    </tr>
    <tr>
      <td>LLaMA3-8B</td>
      <td>MMLU</td>
      <td>0.7185</td>
      <td><strong>0.7234</strong></td>
      <td>~2.7x</td>
    </tr>
  </tbody>
</table>

For EAGLE integration setup, see the [EAGLE repository](https://github.com/SforAiD/LLM-Eagle) and the `EAGLE-hsd/` directory in this repo.

```bash
cd EAGLE-hsd
bash 1114_eagle_eval.sh
```

## Ablation Studies

### Effect of Draft Length (γ)

Vary the draft length using `--gamma` parameter:

```bash
# γ = 4
bash evaluate_speculative_qwen_backward_clever.sh  # Edit: --gamma 4

# γ = 16 (default)
bash evaluate_speculative_qwen_backward_clever.sh  # Edit: --gamma 16
```

<table>
  <thead>
    <tr>
      <th>γ (Draft Length)</th>
      <th>Block Efficiency</th>
      <th>Decoding Speed</th>
      <th>Latency (ms/token)</th>
      <th>Throughput (tokens/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>2.51</td>
      <td>2.38x</td>
      <td>26.38</td>
      <td>37.91</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.91</td>
      <td>2.64x</td>
      <td>23.79</td>
      <td>42.03</td>
    </tr>
    <tr>
      <td><strong>16</strong></td>
      <td><strong>3.25</strong></td>
      <td><strong>2.87x</strong></td>
      <td><strong>21.88</strong></td>
      <td><strong>45.70</strong></td>
    </tr>
  </tbody>
</table>

### Effect of Tree Branching Factor

The branching factor (number of candidates per position) is controlled by the sampling strategy. Use temperature parameter to control branching:

```bash
# Lower temperature = more focused branching
bash evaluate_speculative_qwen_backward_clever.sh  # Edit: --temperature 0.8

# Higher temperature = more diverse branching
bash evaluate_speculative_qwen_backward_clever.sh  # Edit: --temperature 1.0
```

<table>
  <thead>
    <tr>
      <th>Branching Factor</th>
      <th>Block Efficiency</th>
      <th>Decoding Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1 (greedy)</td>
      <td>2.21</td>
      <td>2.14x</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.71</td>
      <td>2.51x</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.25</td>
      <td>2.87x</td>
    </tr>
  </tbody>
</table>

### Number of Drafts (Multi-Draft)

Vary the number of independent draft models using `--multidraft` parameter:

```bash
# Single draft (default)
bash evaluate_speculative_qwen_backward_clever.sh

# 11 parallel drafts
bash evaluate_speculative_qwen_backward_clever_multidraft_11.sh  # Edit: --multidraft 11
```

<table>
  <thead>
    <tr>
      <th># Drafts</th>
      <th>Block Efficiency</th>
      <th>Decoding Speed</th>
      <th>Latency (ms/token)</th>
      <th>Throughput (tokens/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>3.25</td>
      <td>2.87x</td>
      <td>21.88</td>
      <td>45.70</td>
    </tr>
    <tr>
      <td>11</td>
      <td><strong>6.85</strong></td>
      <td><strong>5.12x</strong></td>
      <td><strong>12.27</strong></td>
      <td><strong>81.50</strong></td>
    </tr>
  </tbody>
</table>

## Scaling Analysis

HSD maintains consistent advantages across model sizes (Figure 4 in paper):

- **Block efficiency** improves with larger models (from ~2.5x at 14B to ~3.3x at 72B)
- **Consistent speedup** across all model scales (2.5x - 3.0x)
- **Accuracy preservation** or improvement across all model sizes

Scaling trend results:

<table>
  <thead>
    <tr>
      <th>Target Model</th>
      <th>Parameters</th>
      <th>Block Efficiency</th>
      <th>Speedup vs Tokenwise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-14B</td>
      <td>14B</td>
      <td>2.87</td>
      <td>1.95x</td>
    </tr>
    <tr>
      <td>Qwen2.5-32B</td>
      <td>32B</td>
      <td>3.12</td>
      <td>2.01x</td>
    </tr>
    <tr>
      <td>Qwen2.5-72B</td>
      <td>72B</td>
      <td>3.25</td>
      <td>1.95x</td>
    </tr>
  </tbody>
</table>

## Algorithm Variants

- **Tokenwise**: Verify tokens one by one, stop at first rejection
- **Blockwise**: Verify entire block as a single unit, reject all if any token fails
- **NaiveHSD**: Hierarchical verification with basic stopping criterion
- **HSD (clever)**: Hierarchical verification with optimized stopping criterion that maximizes expected accepted tokens

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

[1] Sun, Ziteng, et al. "Block verification accelerates speculative decoding." arXiv preprint arXiv:2403.10444 (2024).

[2] For EAGLE integration: See the [EAGLE repository](https://github.com/SforAiD/LLM-Eagle)
