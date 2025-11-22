
# # To run gsm8k with HSD:
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path llama-3.1-8b \ # use meta official model is ok too.
  --use_eagle3 \
  --hsd

# # To run gsm8k without HSD:
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_llama3chat \
  --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
  --base-model-path llama-3.1-8b \ # use meta official model is ok too.
  --use_eagle3