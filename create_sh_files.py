import argparse
import copy
import itertools

import numpy as np

from helper import add_training_args, latest_model_path, add_symbreg_args


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
    args.param_name = "boxworld-ribfsqmha-easy"
    args.num_timesteps = int(1e10)
    args.num_checkpoints = 1
    args.seed = 6033
    args.use_wandb = True
    args.wandb_tags = ["easy", "10bn", "grok"]
    args.device = "gpu"
    args.use_valid_env = False
    return args


def write_sh_files(hparams, n_gpu, args):
    keys, values = zip(*hparams.items())
    h_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    h_dict_list = np.random.permutation(h_dict_list)
    arg_list = [copy.deepcopy(args) for _ in h_dict_list]
    n = len(arg_list) // n_gpu
    for gpu in range(n_gpu):
        python_execs = []
        start = gpu * n
        end = (gpu + 1) * n
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=6)
    # parser = add_training_args(parser)
    parser = add_symbreg_args(parser)

    args = parser.parse_args()
    n_gpu = args.n_gpu
    args.__delattr__("n_gpu")

    # args = add_coinrun_sparsity_params(args)
    # args = add_boxworld_params(args)

    hparams = {
        "timeout_in_seconds": [3600 * 10],
        "data_size": [1000, 5000],
        "iterations": [5, 10, 20],
        "n_envs": [32],
        "rounds": [300],
        "denoise": [True, False],
        "populations": [15, 24],
        "procs": [8, 16],
        "ncycles_per_iteration": [550, 1000],
        "bumper": [True, False],
        "binary_operators": [["+", "-", "greater"],
                             # ["+", "-", "greater", "cond", "*"]
                             ],
        "unary_operators": [[],
                            ["sin", "relu"],
                            # ["log", "exp", "relu"],
                            ],
        "logdir": ["logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"]
    }
    n_experiments = np.prod([len(hparams[x]) for x in hparams.keys()])
    print(f"Creating {n_experiments} experiments across {n_gpu} workers.")
    write_sh_files(hparams, n_gpu, args)
