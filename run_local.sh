export PATH=/vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/:/vol/cuda/12.2.0/bin/:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/12.2.0/lib64:/vol/cuda/12.2.0/lib
source /vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/activate
. /vol/cuda/12.2.0/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
export CUDA_DIR=/vol/cuda/12.2.0/:${CUDAPATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/12.2.0/
python3.8 /vol/bitbucket/${USER}/train-procgen-pytorch/train.py --exp_name coinrun --env_name coinrun --num_levels 100000 --distribution_mode hard --param_name hard-500-impalavqmha-rib --num_timesteps 2000000000 --num_checkpoints 200 --seed 6033 --random_percent 0 --use_wandb --real_procgen --device gpu --detect_nan
