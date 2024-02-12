import argparse
import random

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
    args.num_timesteps = 2 * 2 ** 20
    args.num_checkpoints = 1
    args.seed = 6033
    args.num_levels = 10
    args.start_level = 431
    args.use_wandb = True
    args.wandb_tags = ["n_envs", "n_minibatches", "big_bottleneck", "better_info", "with_vel_info"]
    args.device = "gpu"
    args.use_valid_env = False

    n_envs_steps_minib = [[16, 256, 4],
                          [16, 256, 2],
                          [32, 64, 2],
                          [32, 64, 2],
                          [32, 256, 8],
                          [64, 64, 2],
                          [64, 128, 8],
                          [64, 128, 2],
                          [64, 256, 4],
                          [256, 64, 2],
                          [256, 256, 8],
                          [64, 128, 8]]

    for n_envs, n_steps, n_minibatch in n_envs_steps_minib:
        args.n_envs = n_envs
        args.n_steps = n_steps
        args.n_minibatch = n_minibatch
        args.wandb_name = f"{args.n_envs}x{args.n_steps}_{args.n_minibatch}"
        train_ppo(args)

    # for n_envs in [256, 128, 64, 32, 16]:
    #     for n_steps in [256, 128, 64]:
    #         for n_minibatch in args.minibatches:
    #             args.n_envs = n_envs
    #             args.n_steps = n_steps
    #             args.n_minibatch = n_minibatch
    #             args.wandb_name = f"{args.n_envs}x{args.n_steps}_{args.n_minibatch}"
    #             train_ppo(args)

