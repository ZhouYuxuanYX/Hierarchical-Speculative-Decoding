# Hierarchical Speculative Decoding (HSD)

Hierarchical Speculative Decoding is the state-of-the-art lossless verification algorithm for faster LLM inference. HSD organizes draft tokens in a trie structure and performs level-by-level verification, accepting more tokens in expectation than both tokenwise and blockwise verification.

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

## Performance Results

Using Qwen2.5-0.5B as draft model on GSM8K:

### Accuracy Comparison (Table 2)

| Target Model | Tokenwise | HSD (ours) |
|--------------|-----------|------------|
| 72B          | 0.8213    | **0.8517** |
| 32B          | 0.8213    | **0.8479** |
| 14B          | 0.8327    | **0.8327** |

HSD maintains or improves accuracy while being significantly faster.

### Block Efficiency & Speed (Table 3)

| Method        | Block Efficiency | Decoding Speed |
|---------------|------------------|----------------|
| Tokenwise     | 1.00             | 1.47x          |
| Blockwise     | 2.21             | 2.14x          |
| NaiveHSD      | 2.21             | 2.14x          |
| **HSD (ours)**| **3.25**         | **2.87x**      |

HSD achieves **~2x speedup** over blockwise verification while being lossless.

## Algorithm Variants

- **Tokenwise**: Verify tokens one by one, stop at first rejection
- **Blockwise**: Verify entire block as a single unit, reject all if any token fails
- **NaiveHSD**: Hierarchical verification with basic stopping criterion
- **HSD (clever)**: Hierarchical verification with optimized stopping criterion that maximizes expected accepted tokens

## Citation

```bibtex
@inproceedings{hsd2026,
  title={Hierarchical Speculative Decoding for Faster LLM Inference},
  author={...},
  booktitle={ICLR},
  year={2026}
}
```

## References

[1] Sun, Ziteng, et al. "Block verification accelerates speculative decoding." arXiv preprint arXiv:2403.10444 (2024).
