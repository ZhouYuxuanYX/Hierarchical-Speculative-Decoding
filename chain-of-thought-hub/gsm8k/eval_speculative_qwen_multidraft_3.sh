#!bin/bash
cd /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/chain-of-thought-hub-main/gsm8k
echo "speculative"
eval "$(/root/anaconda3/bin/conda shell.bash hook)"                                                                                                                             
source ~/.bashrc
source activate base
conda activate speculative
conda info
pip install optimum
huggingface-cli login --token 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 --add-to-git-credential

python3 eval_speculative_decoding_llm.py --speculative --multidraft 3
