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
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 1024 --n_steps 64 --wandb_name long_1024x64

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 512 --n_steps 128 --wandb_name long_512x128

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 256 --n_steps 256 --wandb_name long_256x256

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 2048 --n_steps 32 --wandb_name long_2048x32

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 2048 --n_steps 64 --wandb_name long_2048x64

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 1024 --n_steps 128 --wandb_name long_1024x128

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 512 --n_steps 256 --wandb_name long_512x256

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
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 1024 --n_steps 64 --wandb_name long_1024x64

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 512 --n_steps 128 --wandb_name long_512x128

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 256 --n_steps 256 --wandb_name long_256x256

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 2048 --n_steps 32 --wandb_name long_2048x32

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 2048 --n_steps 64 --wandb_name long_2048x64

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 1024 --n_steps 128 --wandb_name long_1024x128

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 512 --n_steps 256 --wandb_name long_512x256

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 512 --n_steps 64 --wandb_name long_512x64

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 256 --n_steps 128 --wandb_name long_256x128

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 1024 --n_steps 32 --wandb_name long_1024x32

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 2048 --n_steps 16 --wandb_name long_2048x16

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 48 --n_steps 256 --wandb_name long_48x256

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 32 --n_steps 256 --wandb_name long_32x256

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 32 --n_steps 40 --wandb_name long_32x40

python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name boxworld --env_name boxworld --distribution_mode hard --param_name boxworld-ribmha-easy --num_timesteps 524288 --num_checkpoints 1 --seed 6033 --use_wandb --wandb_tags n_envs long --device gpu --n_envs 512 --n_steps 64 --wandb_name long_512x64
