import argparse
import copy
import subprocess
import traceback

from helper import add_training_args
from train import train_ppo


def format_args(arg):
    output = ""
    d = arg.__dict__
    for var_name in d.keys():
        output += f"--{var_name} {d[var_name]} "
    return output


def executable_train(hparams):
    return f'"cd pyg/train-procgen-pytorch; source venvproc/bin/activate; train.py {hparams}'

    return '; '.join(
        ["hn=$(hostname)",
         "echo ${hn} > ${hn}.txt",
         "export PATH=/vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/:/vol/cuda/12.2.0/bin/:$PATH",
         "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/12.2.0/lib64:/vol/cuda/12.2.0/lib",
         "source /vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/activate",
         ". /vol/cuda/12.2.0/setup.sh",
         "TERM=vt100",
         "/usr/bin/nvidia-smi",
         "export CUDA_DIR=/vol/cuda/12.2.0/:${CUDAPATH}",
         "export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/12.2.0/",
         f"python3.8 /vol/bitbucket/${{USER}}/train-procgen-pytorch/train.py {hparams}",
         ])


if __name__ == '__main__':
    use_subprocesses = True
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    args.exp_name = "coinrun-hparams"
    args.env_name = "coinrun"
    args.distribution_mode = "hard"
    args.param_name = "hard-500-impalafsqmha-sparse"
    args.num_timesteps = 2e7
    args.num_checkpoints = 1
    args.seed = 6033
    args.num_levels = 500
    args.start_level = 0
    args.use_wandb = True
    args.wandb_tags = ["sparsity"]
    args.device = "gpu"
    args.use_valid_env = False
    args.n_minibatch = 32

    # sparsity = [0.04, 0.001]
    sparsity = [0.01, 0.005, 0.0075]
    if not use_subprocesses:
        for sparsity_coef in sparsity:
            args.sparsity_coef = sparsity_coef
            args.wandb_name = f"sparse_{sparsity_coef:.0E}"
            try:
                train_ppo(args)
            except Exception as e:
                print(f"Encountered error during run for {args.wandb_name}:")
                print(traceback.format_exc())
                continue
    else:
        arg_list = [copy.deepcopy(args) for _ in sparsity]
        for arg, sc in zip(arg_list, sparsity):
            arg.sparsity_coef = sc
            arg.wandb_name = f"sparse_{sc:.1E}"
            hparams = format_args(arg)
            cmd = ["./test.sh", executable_train(hparams)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True,
                                 stderr=subprocess.DEVNULL)
