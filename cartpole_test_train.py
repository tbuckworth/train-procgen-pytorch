import argparse

from helper_local import add_training_args
from train import train_ppo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    # args.exp_name = "test"
    # args.env_name = "cartpole"
    # args.n_envs = 2
    # args.param_name = "cartpole"
    # args.num_timesteps = 1000000
    # args.num_checkpoints = 1
    # args.seed = 6033
    # args.use_wandb = False
    # args.use_valid_env = False
    # args.render = True
    
    args.env_name = "mountain_car"
    args.start_level = 0
    args.num_levels = 500
    args.distribution_mode = "easy"
    args.param_name = "mlpmodel"
    args.device = "cpu"
    args.gpu_device = 0
    args.num_timesteps = 200000000
    args.seed = 6033
    args.log_level = 40
    args.num_checkpoints = 1
    args.gamma = 0.95
    args.entropy_coef = 0.02
    args.n_envs = 512
    args.wandb_name = 8964
    args.wandb_tags = ["discrete", "gravity"]
    args.sparsity_coef = 0.0
    args.random_percent = 0
    args.key_penalty = 0
    args.step_penalty = 0
    args.rand_region = 0
    args.num_threads = 8
    # args.no-detect_nan = True
    args.use_valid_env = True
    args.normalize_rew = True
    # args.no-render =
    args.paint_vel_info = True
    args.reduce_duplicate_actions = True
    args.use_wandb = True
    args.real_procgen = True
    # args.no-mirror_env =

    
    train_ppo(args)
