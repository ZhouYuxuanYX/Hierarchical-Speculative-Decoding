#!/usr/bin/env bash
# Chain vs tree, token-wise vs HSD, on the first 100 GSM8K questions, matched settings.
# Both draft 6 tokens deep (chain: 6 drafted tokens; tree: k=4, 6 levels, 85 nodes incl. root) and
# share every other server flag, sampling (T=0.6, top-p 0.9, max 1024 tokens) and the grader.
#   GPU 0: SGLang, concurrency 1 then 64, for each config in SGL_CONFIGS
#   GPU 1: paper's reference code (no SGLang), batch 1, REF_CONFIGS
set -uo pipefail
cd "$(dirname "$0")"; source env.sh > /dev/null 2>&1
export HF_HUB_OFFLINE=1
N=100; OUT=runs/hsd100; mkdir -p $OUT
SGL_CONFIGS=${SGL_CONFIGS:-"chain_token chain_hsd tree_rrsw tree_hsd"}
REF_CONFIGS=${REF_CONFIGS:-"chain_rrsw,chain_hsd,sampled_rrsw,sampled_hsd"}  # ref.jsonl resumes: done pairs are skipped
COMMON="--speculative-num-steps 6 --mem-fraction-static 0.6 --max-running-requests 32"
CHAIN="$COMMON --speculative-eagle-topk 1 --speculative-num-draft-tokens 7"
TREE="$COMMON --speculative-eagle-topk 4 --speculative-num-draft-tokens 85"
declare -A FLAGS=(
  [chain_hsd]="$CHAIN --speculative-use-hsd"
  [tree_hsd]="$TREE --speculative-use-hsd"
  [chain_token]="$CHAIN --speculative-use-rejection-sampling"  # token-wise rejection sampling
  [tree_rrsw]="$TREE --speculative-use-rejection-sampling"     # token-wise on the tree (RRSw)
)

sglang_gpu0() {
  local PORT=30000
  for c in 1 64; do
    for name in $SGL_CONFIGS; do
      # fresh server per run: its accept-length counter covers this run only
      if curl -sf http://127.0.0.1:$PORT/health > /dev/null; then echo "port $PORT busy, abort"; return 1; fi
      local out=$OUT/sglang_c${c}/$name; mkdir -p $out
      GPU=0 PORT=$PORT bash launch_server.sh ${FLAGS[$name]} > $out/server.log 2>&1 &
      for _ in $(seq 240); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
      /data/twang/sgl_eval_venv/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:$PORT/v1 \
        --num-examples $N --num-threads $c --temperature 0.6 --top-p 0.9 --max-tokens 1024 \
        --out-dir $PWD/$out > $out/gsm8k.log 2>&1
      curl -s http://127.0.0.1:$PORT/server_info > $out/server_info.json
      pkill -f -- "[-]-port $PORT"; sleep 15
      echo "[$(date +%T)] sglang c=$c $name: $(grep -m1 'score' $out/gsm8k.log)"
    done
  done
}

ref_gpu1() {
  CUDA_VISIBLE_DEVICES=1 /data/twang/hsd_ref_venv/bin/python ref_gsm8k.py \
    --base meta-llama/Llama-3.1-8B-Instruct --ea lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --configs $REF_CONFIGS --max-questions $N --out $OUT/ref.jsonl >> $OUT/ref.log 2>&1
  echo "[$(date +%T)] reference done"
}

sglang_gpu0 &
ref_gpu1 &
wait
/data/twang/sgl_eval_venv/bin/python score_ref_gsm8k.py $OUT/ref.jsonl --out $OUT/ref_summary.json
echo "[$(date +%T)] all done"
