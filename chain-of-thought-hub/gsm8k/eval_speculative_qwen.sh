#!bin/bash
cd chain-of-thought-hub-main/gsm8k
echo "speculative"
eval "$(/root/anaconda3/bin/conda shell.bash hook)"                                                                                                                             
source ~/.bashrc
source activate base
conda activate speculative
conda info
pip install optimum

python3 eval_speculative_decoding_llm.py --speculative
