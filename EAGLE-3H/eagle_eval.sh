
# # To run gsm8k with HSD:
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --use_eagle3 \
  --question-end 10 \
  --hsd

# # To run gsm8k without HSD:
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path meta-llama/Llama-3.1-8B-Instruct \
  --question-end 10 \
  --use_eagle3

# # To run gsm8k in baseline:
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.baseline \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --bench-name gsm8k \
  --question-end 10 