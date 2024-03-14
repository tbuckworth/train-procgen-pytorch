import argparse
import random
import traceback

import numpy as np

from helper import add_training_args
from inspect_agent import latest_model_path
from train import train_ppo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    args.exp_name = "coinrun-grok"
    args.env_name = "coinrun"
    args.distribution_mode = "hard"
    args.param_name = "hard-500-impalafsqmha"
    args.num_timesteps = 8e7
    args.num_checkpoints = 80
    args.seed = 6033
    args.num_levels = 10
    args.start_level = 431
    args.use_wandb = True
    args.wandb_tags = ["10_levs", "grok"]
    args.device = "gpu"
    args.n_impala_blocks = 3
    args.increasing_lr = True

    args.model_file = latest_model_path("logs/train/coinrun/coinrun/2024-02-12__09-20-09__seed_6033")
    for eps_clip in [0.2, 0.01, 0.1, 0.025, 0.05]:
        args.eps_clip = eps_clip
        args.wandb_name = f"10lev_grok_clip:{eps_clip}"
        train_ppo(args)
