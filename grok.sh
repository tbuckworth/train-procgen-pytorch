#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tfb115

export PATH=/vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/:/vol/cuda/12.2.0/bin/:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/12.2.0/lib64:/vol/cuda/12.2.0/lib
source /vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/activate
. /vol/cuda/12.2.0/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
export CUDA_DIR=/vol/cuda/12.2.0/:${CUDAPATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/12.2.0/
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train_grok.py

