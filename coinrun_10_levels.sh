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
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name coinrun --env_name coinrun --num_levels 10 --distribution_mode hard --start_level 431 --param_name hard-500-$1 --num_timesteps 200000000 --num_checkpoints 20 --seed 6033 --random_percent 0 --use_wandb --real_procgen --device gpu --wandb_name 10_levs
