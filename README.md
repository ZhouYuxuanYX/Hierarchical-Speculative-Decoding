# Hierarchical-Speculative-Decoding
Hierarchical Speculative Decoding is the state-of-the-art verification algorithm.

## Requirements

Please install `python=3.10` and `transformers=4.46.3`, then copy the provided files under the `transformers` directory to the path:
    
```
anaconda3/envs/your_environment_name/lib/python3.10/site-packages/transformers/
``` 

## HSD Running under Chain-of-Thought

```bash
cd chain-of-thought-hub/gsm8k
```

Then run the corresponding sh files:
- Tokenwise
```bash
evaluate_speculative_qwen.sh
```
- Blockwise
```bash
evaluate_speculative_qwen_blockwise.sh
```
- NaiveHSD (ours)
```bash
eval_speculative_qwen_backward.sh
```
- HSD (ours)
```bash
evaluate_speculative_qwen_backward_clever.sh
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






