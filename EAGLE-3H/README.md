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
CUDA_VISIBLE_DEVICES=1 python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --use_eagle3 \
  --bench-name gsm8k \
  --hsd

# 2. EAGLE3 (without HSD)
CUDA_VISIBLE_DEVICES=1 python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --use_eagle3 \
  --bench-name gsm8k
```

### Performance Evaluation

After running the evaluation, compute metrics using:

```bash
python compute_speculative_stats_1.py
```

## Dataset

The evaluation uses the GSM8K test set from EAGLE3 default settings (80 samples)

## Hardware Information

| Device | GPU | GPU Memory | CPUs | CPU Type |
|--------|-----|------------|------|----------|
| H100 | NVIDIA H100-HGX | 80 GB HBM3 | 64 cores | Intel Platinum 8462Y+ (Sapphire Rapids) |
| H200 | NVIDIA H200-HGX | 141 GB HBM3e | 64 cores | Intel Platinum 8562Y+ (Emerald Rapids) |

## Results

### GSM8K Benchmark

| Device | Method | Speed (tokens/s) | Batch Efficiency (BE) |
|--------|--------|------------------|----------------------|
| H100 | EAGLE | 88.10 | 3.22 |
| H100 | HSD | **107.31** | **3.61** |
| H200 | EAGLE | 116.50 | 3.22 |
| H200 | HSD | **135.42** | **3.61** |