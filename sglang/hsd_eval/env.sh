# source this: the runtime environment for SGLang main on this box (driver 570, CUDA 12.8)
source /home/donglab-gpu-local2/anaconda3/etc/profile.d/conda.sh && conda activate sglang_hsd
export LD_LIBRARY_PATH=/data/twang/cuda13_compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}  # CUDA 13 forward-compat libcuda
export CUDA_HOME=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13                # nvcc 13.0
export PATH=$CUDA_HOME/bin:/data/twang/gcc13_bin:$PATH                                 # gcc-13 for C++20 JIT kernels
export HF_HUB_OFFLINE=1
