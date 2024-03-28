import argparse
import copy

from helper import add_training_args, latest_model_path


def format_args(arg):
    output = ""
    d = arg.__dict__
    for var_name in d.keys():
        if d[var_name] is not None:
            if type(d[var_name]) == bool:
                if d[var_name]:
                    output += f"--{var_name} "
            elif type(d[var_name]) == list:
                output += f"--{var_name} {' '.join(d[var_name])} "
            else:
                output += f"--{var_name} {d[var_name]} "
    return output


def executable_train(hparams, name):
    # return f'"hn=$(hostname); echo ${{hn}} > ${{hn}}.txt; cd pyg/train-procgen-pytorch; source venvcartpole/bin/activate; train.py {hparams}"'

    return '\n'.join(
        ["#!/bin/bash",
         "#SBATCH --gres=gpu:1",
         "#SBATCH --mail-type=ALL",
         "#SBATCH --mail-user=tfb115",
         "export PATH=/vol/bitbucket/${USER}/train-procgen-pytorch/venvcartpole/bin/:/vol/cuda/12.2.0/bin/:$PATH",
         "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/12.2.0/lib64:/vol/cuda/12.2.0/lib",
         "source /vol/bitbucket/${USER}/train-procgen-pytorch/venvcartpole/bin/activate",
         ". /vol/cuda/12.2.0/setup.sh",
         "TERM=vt100",
         "/usr/bin/nvidia-smi",
         "export CUDA_DIR=/vol/cuda/12.2.0/:${CUDAPATH}",
         "export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/12.2.0/",
         f"python3.8 /vol/bitbucket/${{USER}}/train-procgen-pytorch/train.py {hparams} 2>&1 | tee /vol/bitbucket/${{USER}}/train-procgen-pytorch/scripts/train_{name}.out\n",
         ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()
    args.exp_name = "coinrun-hparams"
    args.env_name = "coinrun"
    args.distribution_mode = "hard"
    args.param_name = "hard-500-impalafsqmha-sparse"
    args.num_timesteps = int(1e9)
    args.num_checkpoints = 1
    args.seed = 6033
    args.num_levels = 500
    args.start_level = 0
    args.use_wandb = True
    args.wandb_tags = ["sparsity", "fine-tune"]
    args.device = "gpu"
    args.use_valid_env = False
    args.n_minibatch = 16
    args.model_file = latest_model_path("logs/train/coinrun/coinrun/2024-03-26__09-11-40__seed_6033")

    sparsity = [0.05, 0.075, 0.1, 0.2]
    arg_list = [copy.deepcopy(args) for _ in sparsity]
    for arg, sc in zip(arg_list, sparsity):
        arg.sparsity_coef = sc
        arg.wandb_name = f"ft_sparse_{sc:.1E}"
        hparams = format_args(arg)
        exe = executable_train(hparams, arg.wandb_name)
        exe_file_name = f"scripts/tmp_file_{arg.wandb_name}.sh"
        f = open(exe_file_name, 'w', newline='\n')
        f.write(exe)
        f.close()
