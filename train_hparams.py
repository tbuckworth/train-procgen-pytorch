import argparse
import copy
import random
import subprocess
import traceback

from helper import add_training_args
from train import train_ppo


def format_args(arg):
    output = ""
    d = arg.__dict__
    for var_name in d.keys():
        if d[var_name] is not None:
            if type(d[var_name]) == bool:
                if d[var_name]:
                    output += f"--{var_name} "
            else:
                output += f"--{var_name} {d[var_name]} "
    return output


def executable_train(hparams, name):
    # return f'"hn=$(hostname); echo ${{hn}} > ${{hn}}.txt; cd pyg/train-procgen-pytorch; source venvproc/bin/activate; train.py {hparams}"'

    return '\n'.join(
        ["hn=$(hostname)",
         "echo ${hn} > ${hn}.txt",
         "source /vol/bitbucket/${USER}/train-procgen-pytorch/venvproc/bin/activate",
         f"nohup python3.8 /vol/bitbucket/${{USER}}/train-procgen-pytorch/train.py {hparams} &\n",
         # f"2>&1 | tee /vol/bitbucket/${{USER}}/train-procgen-pytorch/scripts/train_{name}.out\n",
         ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_process', action="store_true", default=False)
    parser = add_training_args(parser)
    args = parser.parse_args()
    single_process = args.single_process
    args.__delattr__("single_process")
    args.exp_name = "coinrun-hparams"
    args.env_name = "coinrun"
    args.distribution_mode = "hard"
    args.param_name = "hard-500-impalafsqmha-sparse"
    args.num_timesteps = int(2e7)
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
    if single_process:
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
            n = random.randint(0, 10000)
            hparams = format_args(arg)
            exe = executable_train(hparams, n)
            exe_file_name = f"scripts/tmp_file_{n}.sh"
            f = open(exe_file_name, 'w', newline='\n')
            f.write(exe)
            f.close()
            cmd = ["./test.sh", exe_file_name]  # executable_train(hparams)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True,
                                 stderr=subprocess.DEVNULL)
