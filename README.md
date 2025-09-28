# Hierarchical-Speculative-Decoding
Hierarchical Speculative Decoding is the state-of-the-art lossless verification algorithm.

## HSD Running under Chain-of-Thought
Baseline (no HSD):
```bash
python3 eval_speculative_decoding_llm.py --speculative --naive
```
Ours (with HSD):
```bash
python3 eval_speculative_decoding_llm.py --speculative --HSD
```



