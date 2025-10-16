# Hierarchical-Speculative-Decoding
Hierarchical Speculative Decoding is the state-of-the-art verification algorithm.

## Requirements

Please install `python=3.10` and `transformers=4.46.3`, then copy the provided files under the `transformers` directory to the path:
    
```
anaconda3/envs/your_environment_name/lib/python3.10/site-packages/transformers/
``` 

## Reproduce the Experiment Results on GSM8K

```bash
cd chain-of-thought-hub/gsm8k
```
### Singledraft Verification
Please run the corresponding shell files:

- Tokenwise
```bash
bash evaluate_speculative_qwen.sh
```
- Blockwise
```bash
bash evaluate_speculative_qwen_blockwise.sh
```
- NaiveHSD (ours)
```bash
bash eval_speculative_qwen_backward.sh
```
- HSD (ours)
```bash
bash evaluate_speculative_qwen_backward_clever.sh
```
### Multidraft Verification with Recursive Reject Sampling
Please run the corresponding shell files:

- Tokenwise
```bash
bash evaluate_speculative_qwen_multidraft_11.sh
```
- HSD (ours)
```bash
bash evaluate_speculative_qwen_backward_clever_multidraft_11.sh
```

Finally, evaluate Block Efficiency and Decoding Speed by running the following command:

```bash
python compute_speculative_stats.py
```


## Performance Comparison Across Model Sizes and Methods

| Metric | Method | 72B | 32B | 14B |
|--------|--------|-----|-----|-----|
| GSM8K (Accuracy) | Tokenwise | 0.8213 | 0.8213 | 0.8327 |
| GSM8K (Accuracy) | HSD | 0.8517 | 0.8479 | 0.8327 |

Table 1. Comparison of GSM8K performance of Tokenwise verification and HSD (ours) using Qwen2.5-0.5B as draft model.






