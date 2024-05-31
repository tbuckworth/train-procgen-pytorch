import os
import argparse
import copy
import itertools
import time

import numpy as np
import re
from helper_local import latest_model_path, run_subprocess, DictToArgs, free_gpu, get_config


def format_args(arg):
    output = ""
    d = arg.__dict__
    for var_name in d.keys():
        v = d[var_name]
        if v is not None:
            if type(v) == bool:
                if v:
                    output += f"--{var_name} "
                else:
                    output += f"--no-{var_name} "
            elif type(v) == list:
                if len(v) > 0:
                    if type(v[0]) == int:
                        v = [str(x) for x in v]
                    output += f"--{var_name} {' '.join(v)} "
            else:
                output += f"--{var_name} {v} "
    return output


def executable_python(hparams, name, script="train"):
    return f"python3.8 /vol/bitbucket/${{USER}}/train-procgen-pytorch/{script}.py {hparams} 2>&1 | tee /vol/bitbucket/${{USER}}/train-procgen-pytorch/scripts/train_{name}.out\n"


def executable_train(python_execs=None, slurm=False):
    if python_execs is None:
        python_execs = []
    if not slurm:
        return '\n'.join(
            ["#!/bin/bash",
             "source /vol/bitbucket/${USER}/train-procgen-pytorch/venvcartpole/bin/activate",
             ] + python_execs + ["exit", "exit"])
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


def write_sh_files(hparams, n_gpu, args, execute, cuda, random_subset, hparam_type, re_use_machine=False,
                   specify_host=None, compute_all=True, slurm=False):
    hosts = {}
    free_machine = None
    keys, values = zip(*hparams.items())
    if not compute_all:
        h_dict_list = []
        for key in keys:
            if len(hparams[key]) > 1:
                for param in hparams[key]:
                    temp_dict = {}
                    temp_dict[key] = param
                    for k2 in keys:
                        if k2 != key:
                            hs = middle(hparams[k2])
                            if type(hs) is list:
                                hs = hs.copy()
                            temp_dict[k2] = hs
                    if "wandb_tags" in keys:
                        temp_dict["wandb_tags"] += [f"varied-{key}"]
                    h_dict_list.append(temp_dict.copy())
                    del temp_dict
    else:
        h_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    h_dict_list = np.random.permutation(h_dict_list)
    arg_list = [copy.deepcopy(args) for _ in h_dict_list]

    n_execs = len(arg_list)
    if n_execs < n_gpu:
        n_gpu = n_execs
    n = n_execs // n_gpu
    for gpu in range(n_gpu):
        python_execs = []
        start = gpu * n
        end = (gpu + 1) * n
        for arg, h_dict in zip(arg_list[start:end], h_dict_list[start:end]):
            # for hparam_list in permutations_dicts[gpu * n:(gpu + 1) * n]:
            nme = np.random.randint(0, 10000)
            # nme = ""
            for key in h_dict.keys():
                v = h_dict[key]
                v_str = v
                if type(v) == list:
                    v_str = ','.join([str(x) for x in v])
                # nme += f"_{key}_{v_str}"
                arg.__dict__[key] = v
            arg.wandb_name = nme
            hparams = format_args(arg)
            if hparam_type == "symbreg":
                script = "symbolic_regression"
            if hparam_type == "train":
                script = "train"
            python_execs += [executable_python(hparams, arg.wandb_name, script)]
        cut_to = int(random_subset * len(python_execs))
        python_execs = list(np.random.choice(python_execs, cut_to, replace=False))
        exe = executable_train(python_execs, slurm)
        exe_file_name = f"scripts/tmp_file_{arg.wandb_name}.sh"
        f = open(exe_file_name, 'w', newline='\n')
        f.write(exe)
        f.close()
        session_name = f"tmpSession{np.random.randint(0, 1000)}"
        if slurm:
            # # command = f"'cd pyg/train-procgen-pytorch\n sbatch {exe_file_name}'"
            # cmd1 = f'ssh gpucluster2 "sbatch pyg/train-procgen-pytorch/{exe_file_name}"'
            # run_subprocess(cmd1, "\\n", suppress=False)
            #

            command = f"'cd pyg/train-procgen-pytorch\n sbatch {exe_file_name}'"

            cmd1 = f'ssh gpucluster2 "tmux new -d -s {session_name}"'
            cmd2 = f'ssh gpucluster2 "tmux send -t {session_name}.0 {command} ENTER"'
            cmd3 = f'ssh gpucluster2 "tmux send -t {session_name}.0 exit"'
            for cmd in [cmd1, cmd2, cmd3]:
               run_subprocess(cmd, "\\n", suppress=False)

        elif execute:
            script = "~/free_cpu"
            if cuda:
                script = "~/free_gpu"
            found = False
            for attempts in range(30):
                if specify_host is None:
                    if cuda:
                        if not re_use_machine or free_machine is None:
                            free_machine = free_gpu(hosts)
                    else:
                        free_machine = run_subprocess(script, "\\n", suppress=True)
                    host = re.search(r"(.*).doc.ic.ac.uk", free_machine).group(1)
                else:
                    host = specify_host
                if host not in hosts.keys():
                    hosts[host] = [session_name]
                    found = True
                    break
            if not found:
                hosts[host].append(session_name)

            command = f"'cd pyg/train-procgen-pytorch\n source {exe_file_name}'"

            print(f"Host:{host}\tSessionName:{session_name}\tScript:{exe_file_name}")

            cmd1 = f'ssh {host} "tmux new -d -s {session_name}"'
            cmd2 = f'ssh {host} "tmux send -t {session_name}.0 {command} ENTER"'

            run_subprocess(cmd1, "\\n", suppress=False)
            run_subprocess(cmd2, "\\n", suppress=False)
    np.save(os.path.join("data", f"hosts_{time.strftime('%Y-%m-%d__%H-%M-%S')}.npy"), hosts)


def middle(hs):
    hs.sort()
    return hs[len(hs) // 2]


def symbreg_hparams():
    return {
        "timeout_in_seconds": [3600 * 10],
        "data_size": [2000, 1000, 5000],  # , 500, 100, 50],# 5000],
        "iterations": [1, 5, 20],  # 20, 40, 80],
        "n_envs": [100],
        "rounds": [1000],
        "denoise": [True],
        "populations": [24],
        "procs": [8],
        "ncycles_per_iteration": [4000],
        "bumper": [False],
        "binary_operators": [["+", "-", "greater", "\*", "/"]],  # "cond"
        "unary_operators": [  # [],
            ["sin", "relu", "log", "exp", "sign", "sqrt", "square"],
        ],
        "wandb_tags": [["cartpole", "hparam", "multi-extrapolation", "cartpole_swing"]],
        "model_selection": ["accuracy", "best"],
        "weight_metric": ["value", "entropy"],
        # "loss_function": ["capped_sigmoid"],
        # "loss_function": ['sigmoid', 'exp', 'logitmarg', 'logitdist', 'mse', 'capped_sigmoid'],
        "loss_function": ['mse', 'capped_sigmoid', "logitdist", "sigmoid", "mce"],  # 'exp', 'sigmoid', 'logitdist', ],
        # "logdir": ["logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"],
        # "logdir": ["logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"],

        # high-entropy one:
        # "logdir": ["logs/train/boxworld/boxworld/2024-04-08__14-52-30__seed_6033"],
        # "logdir": ["logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"],
        # "logdir": ["logs/train/coinrun/coinrun-hparams/2024-04-18__08-38-17__seed_6033"],

        ## Actually working ones:
        "logdir": [
            "logs/train/mountain_car/test/2024-05-22__20-29-39__seed_30",
            # # "logs/train/acrobot/test/2024-05-01__12-22-24__seed_6033",
            # # "logs/train/mountain_car/test/2024-05-03__15-46-58__seed_6033",
            # # Best generalizer:
            # "logs/train/cartpole_swing/test/2024-05-01__14-19-53__seed_6033",
            # # low test ep reward and length:
            # "logs/train/cartpole_swing/test/2024-05-01__14-19-59__seed_6033",
            # # low test ep reward, high length:
            # "logs/train/cartpole_swing/test/2024-05-01__14-19-58__seed_6033",

        ],
        # "logs/train/cartpole/test/2024-05-01__11-17-14__seed_0",
        # "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033"],

        "use_wandb": [True],
        "stochastic": [False],
    }


def mountain_car_hparams():
    return {
        "exp_name": [None],
        "env_name": ['mountain_car'],
        # "distribution_mode": ['hard'],
        "param_name": ['mlpmodel'],
        "device": ["gpu"],
        "num_timesteps": [int(1e6)],
        "seed": [6033, 0, 4, 5, 1, 42, 30],
        "gamma": [0.99],  # [0.95],# 0.99],
        "lmbda": [0.98],
        "learning_rate": [0.0003],  # [0.0005],# 0.0003],
        "entropy_coef": [0],  # [0.02],# 0],
        "n_envs": [16],  # [512, 1024],
        "n_steps": [16],  # [256],
        "n_epochs": [4],  # [3],# 4],
        # "n_minibatch": None,
        # "mini_batch_size": None,
        # "wandb_name": None,
        # "wandb_group": None,
        "wandb_tags": [["lesser gravity", "sb3"]],
        # "detect_nan": False,
        "use_wandb": [True],
        "mirror_env": [False],
        "use_valid_env": [True],
        # "output_dim": [256, 1, 9],
        # "fs_coef": [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }


def acrobot_hparams():
    return {
        "exp_name": [None],
        "env_name": ['acrobot'],
        # "distribution_mode": ['hard'],
        "param_name": ['mlpmodel'],
        "device": ["gpu"],
        "num_timesteps": [int(5e7)],
        "seed": [6033, 0, 1, 101, 40],
        "gamma": [0.95],
        "learning_rate": [0.0005],
        "entropy_coef": [0.02],
        "n_envs": [256],
        "n_steps": [256],
        "n_minibatch": [16],
        # "mini_batch_size": None,
        # "wandb_name": None,
        # "wandb_group": None,
        "wandb_tags": [["multi-extrapolation", "gravity", "fixed"]],
        # "detect_nan": False,
        "use_wandb": [True],
        "mirror_env": [False],
        "use_valid_env": [True],
        # "output_dim": [256, 1, 9],
        # "fs_coef": [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }


def cartpole_hparams_transformer():
    return {
        "exp_name": [None],
        "env_name": ['cartpole'],
        # "distribution_mode": ['hard'],
        "param_name": ['cartpole_transform'],
        "device": ["gpu"],
        "num_timesteps": [int(2e8)],
        "seed": [6033],  # 0, 1, 101, 40],
        "gamma": [0.95, 0.9, 0.99],
        "learning_rate": [0.0005, 0.001, 0.00025, 0.01],
        "entropy_coef": [0, 0.01, 0.02, 0.05],
        "n_envs": [1024, 512, 2048],
        "n_steps": [256, 128],
        # "n_minibatch": None,
        # "mini_batch_size": None,
        # "wandb_name": None,
        # "wandb_group": None,
        "wandb_tags": [["multi-extrapolation"]],
        # "detect_nan": False,
        "use_wandb": [True],
        "mirror_env": [False],
        "use_valid_env": [True],
        # "output_dim": [256, 1, 9],
        # "fs_coef": [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }


def cartpole_hparams():
    return {
        "exp_name": [None],
        "env_name": ['cartpole_swing'],
        # "distribution_mode": ['hard'],
        "param_name": ['mlpmodel'],
        "device": ["gpu"],
        "num_timesteps": [int(2e8)],
        "seed": [6033],  # 0, 1, 101, 40],
        "gamma": [0.95, 0.9],
        "learning_rate": [0.0005, 0.001, 0.00025, 0.01],
        "entropy_coef": [0, 0.02],
        "n_envs": [1024, 512],
        "n_steps": [256],
        # "n_minibatch": None,
        # "mini_batch_size": None,
        # "wandb_name": None,
        # "wandb_group": None,
        "wandb_tags": [["multi-extrapolation", "fixed_reward"]],
        # "detect_nan": False,
        "use_wandb": [True],
        "mirror_env": [False],
        "use_valid_env": [True],
        # "output_dim": [256, 1, 9],
        # "fs_coef": [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }


def coinrun_mostlyneural_hparams():
    return {
        "exp_name": ['coinrun-hparams'],
        "env_name": ['coinrun'],
        # "distribution_mode": ['hard'],
        "param_name": ['hard-500-impala'],
        "device": ["gpu"],
        "num_timesteps": [int(20e8)],
        "seed": [6033],
        # "n_minibatch": None,
        # "mini_batch_size": None,
        # "wandb_name": None,
        # "wandb_group": None,
        "wandb_tags": [["mostly-neural", "impala"]],
        # "detect_nan": False,
        "use_wandb": [True],
        "mirror_env": [False],
        "use_valid_env": [True],
        "output_dim": [12, 24],
    }


def cartpole_graph_transition_hparams():
    return {
        "exp_name": [None],
        "env_name": ['cartpole'],  # 'cartpole-swing'],
        # "distribution_mode": ['hard'],
        "param_name": ['graph-transition'],
        "device": ["gpu"],
        "num_timesteps": [int(2e7)],
        "seed": [6033],  # 0, 1, 101, 40],
        "gamma": [0.95],  # 0.9],
        "val_epochs": [15, 1, 20, 2, 9],
        "dyn_epochs": [3],
        "learning_rate": [0.0001],  # 0.00025, 0.0005],
        "t_learning_rate": [0.0005],  # 0.00025, 0.0001],
        "n_envs": [64],
        "n_steps": [256],
        "n_rollouts": [3],
        "temperature": [1e-2],  # [0.01, 0.001, 0.0001, 0.00001, 0.000001],
        "use_gae": [True],
        "rew_coef": [1.],  # 10, 1, 0.1],
        "done_coef": [1.],  # 10, 1.], #0.1 bad
        "clip_value": [False],
        # "n_minibatch": None,
        # "mini_batch_size": None,
        # "wandb_name": None,
        # "wandb_group": None,
        "wandb_tags": [["graph-transition", "0.01-temp", "val_epochs"]],
        # "detect_nan": False,
        "use_wandb": [True],
        "mirror_env": [False],
        "use_valid_env": [True],
        "output_dim": [24],
        "anneal_temp": [False],
        # "fs_coef": [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }


def train_hparams():
    # return continue_run("logs/train/mountain_car/test/2024-04-25__05-18-36__seed_6033")
    return cartpole_graph_transition_hparams()


def continue_run(logdir):
    model_file = latest_model_path(logdir)
    cfg = get_config(logdir)
    # hp_file = os.path.join(logdir, "hyperparameters.npy")
    # if os.path.exists(hp_file):
    #     hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    cfg["model_file"] = model_file
    cfg["learning_rate"] = 0.0001

    # May need to change this:
    exclusions = ["algo", "epoch", "lmbda", "grad_clip_norm", "eps_clip", "value_coef", "normalize_adv", "use_gae",
                  "architecture", "recurrent",
                  "no-recurrent", "depth", "latent_size", "mid_weight"]
    cfg = {k: [v] for k, v in cfg.items() if k not in exclusions}

    return cfg


def add_symbreg_args_dict():
    return {
        "data_size": 100,
        "iterations": 1,
        "logdir": None,
        "n_envs": int(8),
        "rounds": int(10),
        "binary_operators": ["+", "-", "greater"],
        "unary_operators": [],
        "denoise": False,
        "use_wandb": False,
        "wandb_tags": [],
        "wandb_name": None,
        "wandb_group": None,
        "timeout_in_seconds": 36000,
        "populations": 24,
        "procs": 8,
        "ncycles_per_iteration": 550,
        "bumper": False,
        "model_selection": "best",
        "stochastic": True,
        "loss_function": "mse",
        "weight_metric": "value",
    }


def add_training_args_dict():
    return {
        "exp_name": 'test',
        "env_name": 'coinrun',
        "val_env_name": None,
        "start_level": int(0),
        "num_levels": int(500),
        "distribution_mode": 'easy',
        "param_name": 'easy-200',
        "device": 'gpu',
        "gpu_device": int(0),
        "num_timesteps": int(25000000),
        "seed": 6033,
        "log_level": int(40),
        "num_checkpoints": int(10),
        "model_file": None,
        "mut_info_alpha": None,
        "gamma": None,
        "lmbda": None,
        "learning_rate": None,
        "entropy_coef": None,
        "n_envs": None,
        "n_steps": None,
        "n_minibatch": None,
        "mini_batch_size": None,
        "wandb_name": None,
        "wandb_group": None,
        "wandb_tags": [],
        "levels": None,
        "sparsity_coef": 0.,

        "random_percent": 0,
        "key_penalty": 0,
        "step_penalty": 0,
        "rand_region": 0,
        "num_threads": 8,

        "detect_nan": False,
        "use_valid_env": True,
        "normalize_rew": True,
        "render": False,
        "paint_vel_info": True,
        "reduce_duplicate_actions": True,
        "use_wandb": False,
        "real_procgen": True,
        "mirror_env": False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=5)
    parser.add_argument('--execute', action="store_true", default=True)
    # parser.add_argument('--cuda', action="store_true", default=False)
    parser.add_argument('--max_runs', type=int, default=200)
    parser.add_argument('--hparam_type', type=str, default="train")
    parser.add_argument('--host', type=str, default=None)  # "gpu20")
    parser.add_argument('--compute_all', action="store_true", default=False)
    parser.add_argument('--re_use_machine', action="store_true", default=False)

    largs = parser.parse_args()

    compute_all = largs.compute_all
    re_use_machine = largs.re_use_machine
    specify_host = largs.host
    if specify_host is not None and not re_use_machine:
        print("Warning - specifying host will re-use that host")

    slurm = False
    if specify_host == "slurm":
        slurm = True

    n_gpu = largs.n_gpu
    execute = largs.execute
    max_runs = largs.max_runs
    hparam_type = largs.hparam_type
    assert hparam_type in ["symbreg", "train"], "hparam_type must be one of 'symbreg', 'train'."

    parser_sub = argparse.ArgumentParser()

    if hparam_type == "train":
        parser_dict = add_training_args_dict()
        hparams = train_hparams()
        cuda = True

    if hparam_type == "symbreg":
        parser_dict = add_symbreg_args_dict()
        hparams = symbreg_hparams()
        cuda = False
    args = DictToArgs(parser_dict)

    # args = add_coinrun_sparsity_params(args)
    # args = add_boxworld_params(args)
    if compute_all:
        n_experiments = np.prod([len(hparams[x]) for x in hparams.keys()])
    else:
        n_experiments = np.sum([len(hparams[x]) for x in hparams.keys()])
    print(f"Creating {n_experiments} experiments across {n_gpu} workers.")
    random_subset = min(1, max_runs / n_experiments)
    write_sh_files(hparams, n_gpu, args, execute, cuda, random_subset, hparam_type, re_use_machine, specify_host,
                   compute_all, slurm)
