#!/bin/bash
# GPU 0 queue: wait for the stock-tree c=64 controls, then chain at depth 6 (batch 1), then reference shard 1; then score.
cd "$(dirname "$0")"
while pgrep -f "[r]un_gsm8k_c64.sh" > /dev/null; do sleep 30; done
sleep 20
GPU=0 PORT=30000 bash run_gsm8k_bs1.sh chain_hsd6 >> runs/gsm8k_bs1_gpu0.log 2>&1
source env.sh > /dev/null 2>&1; export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 /data/twang/hsd_ref_venv/bin/python ref_gsm8k.py --base meta-llama/Llama-3.1-8B-Instruct \
  --ea lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B --num-shards 2 --shard 1 --out runs/ref_gsm8k/shard1.jsonl > runs/ref_gsm8k/shard1.log 2>&1
while pgrep -f "[r]ef_gsm8k.py.*--shard 0" > /dev/null; do sleep 30; done
/data/twang/sgl_eval_venv/bin/python score_ref_gsm8k.py runs/ref_gsm8k/shard0.jsonl runs/ref_gsm8k/shard1.jsonl --out runs/ref_gsm8k/summary.json
