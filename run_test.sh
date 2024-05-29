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
source /vol/bitbucket/${USER}/train-procgen-pytorch/venvcartpole/bin/activate
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --env_name cartpole --start_level 0 --num_levels 500 --distribution_mode easy --param_name graph-transition --device gpu --gpu_device 0 --num_timesteps 4000000 --seed 6033 --log_level 40 --num_checkpoints 10 --gamma 0.95 --learning_rate 0.00025 --n_envs 32 --n_steps 256 --wandb_name 1994 --wandb_tags graph-transition large-output-dim added-cont --sparsity_coef 0.0 --random_percent 0 --key_penalty 0 --step_penalty 0 --rand_region 0 --num_threads 8 --no-detect_nan --use_valid_env --normalize_rew --no-render --paint_vel_info --reduce_duplicate_actions --use_wandb --real_procgen --no-mirror_env --val_epochs 8 --dyn_epochs 5 --t_learning_rate 0.00025 --n_rollouts 3 --temperature 0.01 --use_gae --rew_coef 1 --done_coef 1.0 --no-clip_value --output_dim 24 --anneal_temp
