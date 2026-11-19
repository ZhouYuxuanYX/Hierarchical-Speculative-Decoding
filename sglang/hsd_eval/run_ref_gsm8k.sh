#!/bin/bash
# GSM8K with the paper's reference code (no SGLang), 2 shards: GPU 1 now, GPU 0 after the SGLang batch-1 run.
cd "$(dirname "$0")"
source env.sh > /dev/null 2>&1
export HF_HUB_OFFLINE=1
PY=/data/twang/hsd_ref_venv/bin/python
ARGS="--base meta-llama/Llama-3.1-8B-Instruct --ea lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B --num-shards 2"
mkdir -p runs/ref_gsm8k
CUDA_VISIBLE_DEVICES=1 $PY ref_gsm8k.py $ARGS --shard 0 --out runs/ref_gsm8k/shard0.jsonl > runs/ref_gsm8k/shard0.log 2>&1 &
until grep -q "tree_hsd done\|tree_hsd.*abort" runs/gsm8k_bs1_gpu0.log; do sleep 30; done
sleep 30
CUDA_VISIBLE_DEVICES=0 $PY ref_gsm8k.py $ARGS --shard 1 --out runs/ref_gsm8k/shard1.jsonl > runs/ref_gsm8k/shard1.log 2>&1 &
wait
/data/twang/sgl_eval_venv/bin/python score_ref_gsm8k.py runs/ref_gsm8k/shard0.jsonl runs/ref_gsm8k/shard1.jsonl --out runs/ref_gsm8k/summary.json
