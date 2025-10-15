# Hierarchical-Speculative-Decoding
Hierarchical Speculative Decoding is the state-of-the-art verification algorithm.

## HSD Running under Chain-of-Thought
Tokenwise (Baseline):
```bash
python3 eval_speculative_decoding_llm.py --speculative 
```
HSD (Ours):
```bash
python3 eval_speculative_decoding_llm.py --speculative --HSD
```

## Performance Comparison Across Model Sizes and Methods

| Metric | Method | 72B | 32B | 14B |
|--------|--------|-----|-----|-----|
| GSM8K (Accuracy) | Tokenwise | 0.8213 | 0.8213 | 0.8327 |
| GSM8K (Accuracy) | HSD | 0.8517 | 0.8479 | 0.8327 |



