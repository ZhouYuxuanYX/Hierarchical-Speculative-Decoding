#!bin/bash
cd /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/chain-of-thought-hub-main/gsm8k
echo "speculative"
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
source ~/.bashrc
source activate base
conda activate speculative
conda info
python3 eval_speculative_decoding_llm.py --backward --speculative --recursive
