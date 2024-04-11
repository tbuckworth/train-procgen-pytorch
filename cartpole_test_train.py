import argparse

from helper_local import add_training_args
from train import train_ppo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    args.exp_name = "test"
    args.env_name = "cartpole"
    args.n_envs = 2
    args.param_name = "cartpole"
    args.num_timesteps = 1000000
    args.num_checkpoints = 1
    args.seed = 6033
    args.use_wandb = False
    args.use_valid_env = False
    args.render = True
    train_ppo(args)
