import argparse
import traceback

from helper import add_training_args
from train import train_ppo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    args.exp_name = "coinrun-hparams"
    args.env_name = "coinrun"
    args.distribution_mode = "hard"
    args.param_name = "hard-500-impalafsqmha"
    args.num_timesteps = 2e7
    args.num_checkpoints = 1
    args.seed = 6033
    args.num_levels = 500
    args.start_level = 0
    args.use_wandb = True
    args.wandb_tags = ["sparsity"]
    args.device = "gpu"
    args.use_valid_env = False
    args.n_minibatch = 16

    sparsity = [0.04, 0.001]
    sparsity = [0.02, 0.002, 0.01, 0.005, 0.0075]

    for sparsity_coef in sparsity:
        args.sparsity_coef = sparsity_coef
        args.wandb_name = f"sparse_{sparsity_coef:.0E}"
        try:
            train_ppo(args)
        except Exception as e:
            print(f"Encountered error during run for {args.wandb_name}:")
            print(traceback.format_exc())
            continue
