import argparse
import copy
import itertools

import numpy as np

from helper import add_training_args, latest_model_path


def format_args(arg):
    output = ""
    d = arg.__dict__
    for var_name in d.keys():
        v = d[var_name]
        if v is not None:
            if type(v) == bool:
                if v:
                    output += f"--{var_name} "
            elif type(v) == list:
                if len(v) > 0 and type(v[0]) == int:
                    v = [str(x) for x in v]
                output += f"--{var_name} {' '.join(v)} "
            else:
                output += f"--{var_name} {v} "
    return output


def executable_python(hparams, name):
    return f"python3.8 /vol/bitbucket/${{USER}}/train-procgen-pytorch/train.py {hparams} 2>&1 | tee /vol/bitbucket/${{USER}}/train-procgen-pytorch/scripts/train_{name}.out\n"


def executable_train(hparams, name, python_execs=[]):
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
         ] + python_execs)


def add_coinrun_sparsity_params(args):
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
    return args


def add_boxworld_params(args):
    args.exp_name = "boxworld"
    args.env_name = "boxworld"
    args.param_name = "boxworld-ribfsqmha"
    args.num_timesteps = int(2e8)
    args.num_checkpoints = 1
    args.seed = 6033
    args.use_wandb = True
    args.wandb_tags = ["h-param", "hard", "n_envs", "n_minibatch"]
    args.device = "gpu"
    args.use_valid_env = False
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()
    n_gpu = 6

    # args = add_coinrun_sparsity_params(args)
    args = add_boxworld_params(args)

    hparams = {
        "n_envs": [1024, 2048],
        "gamma": [0.95],
        "n_minibatch": [16],
        "normalize_rew": [True],
        "levels": [[10, 10]],#, [8, 5, 5, 5], [9, 9, 9]],
        "learning_rate": [0.0025, 0.001, 0.0005],
        "entropy_coef": [0.001, 0.01, 0.02, 0.005, 0.0005]
    }
    # h_dict = [v for v in itertools.product(*hparams.values())]
    keys, values = zip(*hparams.items())
    h_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    h_dict_list = np.random.permutation(h_dict_list)

    arg_list = [copy.deepcopy(args) for _ in h_dict_list]
    n = len(arg_list) // n_gpu
    for gpu in range(n_gpu):
        python_execs = []
        start = gpu*n
        end = (gpu+1)*n
        for arg, h_dict in zip(arg_list[start:end], h_dict_list[start:end]):
            # for hparam_list in permutations_dicts[gpu * n:(gpu + 1) * n]:
            nme = ""
            for key in h_dict.keys():
                v = h_dict[key]
                v_str = v
                if type(v) == list:
                    v_str = ','.join([str(x) for x in v])
                nme += f"_{key}_{v_str}"
                arg.__dict__[key] = v
            arg.wandb_name = nme
            hparams = format_args(arg)
            python_execs += [executable_python(hparams, arg.wandb_name)]
        exe = executable_train(hparams, arg.wandb_name, python_execs)
        exe_file_name = f"scripts/tmp_file_{arg.wandb_name}.sh"
        f = open(exe_file_name, 'w', newline='\n')
        f.write(exe)
        f.close()
