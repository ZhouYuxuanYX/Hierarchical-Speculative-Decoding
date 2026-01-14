# EAGLE-3H: EAGLE3 with Hierarchical Speculative Decoding

This repository evaluates EAGLE3 with Hierarchical Speculative Decoding (HSD) on the GSM8K benchmark.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Running Evaluations

The `eagle_eval.sh` script provides three evaluation modes:

```bash
# 1. EAGLE3 with HSD (Hierarchical Speculative Decoding)
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --use_eagle3 \
  --hsd

# 2. EAGLE3 (without HSD)
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --use_eagle3

# 3. Baseline (autoregressive decoding)
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.baseline \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --bench-name gsm8k
```

### Performance Evaluation

After running the evaluation, compute metrics using:

```bash
python performance_evaluation.py --answer-file /path/to/output/llama38b2_40-temperature-1.0_ea_hsd.jsonl
```

## Dataset

The evaluation uses the full GSM8K test set (1319 samples) loaded from HuggingFace: `openai/gsm8k`.

