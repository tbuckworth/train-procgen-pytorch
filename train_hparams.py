import argparse
import random

import numpy as np

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
    args.wandb_tags = ["bottleneck_search"]
    args.device = "gpu"
    args.use_valid_env = False
    args.n_minibatch = 16

    # hparams = [
    #     [3, [8, 5, 5, 5]],
    #     [4, [8, 5, 5, 5]],
    #     [3, [4, 5, 5, 5]],
    #     [4, [4, 5, 5, 5]],
    #     [3, [10, 10]],
    #     [4, [10, 10]],
    # ]
    impala_blocks = [4, 3]
    level_list = [
        [1, 1],
        [2, 2],
        [4, 4],
        [8, 8],
        [1, 1, 1],
        [2, 2, 2],
        [3, 2, 2],
        [3, 3, 2]
    ]



    envs = [
        [0, 500, ["bottleneck_search", "500 levs"]],
        # [431, 10, ["bottleneck_search", "10 levs"]],
    ]

    for start_level, num_levels, tags in envs:
        args.start_level = start_level
        args.num_levels = num_levels
        args.wandb_tags = tags
        # for n_impala_blocks, levels in hparams:
        for n_impala_blocks in impala_blocks:
            for levels in level_list:
                args.n_impala_blocks = n_impala_blocks
                args.levels = levels
                args.codebook_size = np.prod(levels)
                size = 64//2**n_impala_blocks
                args.latent_size = size
                args.wandb_name = f"{n_impala_blocks}({size}x{size})x{','.join([str(x) for x in levels])}_({np.prod(levels)})"
                train_ppo(args)
                # try:
                #     train_ppo(args)
                # except Exception as e:
                #     print(f"Encountered error during run for {args.wandb_name}:")
                #     print(e)
                #     continue

    # for n_envs in [256, 128, 64, 32, 16]:
    #     for n_steps in [256, 128, 64]:
    #         for n_minibatch in args.minibatches:
    #             args.n_envs = n_envs
    #             args.n_steps = n_steps
    #             args.n_minibatch = n_minibatch
    #             args.wandb_name = f"{args.n_envs}x{args.n_steps}_{args.n_minibatch}"
    #             train_ppo(args)

