import argparse

from helper_local import add_training_args, latest_model_path, get_config, DictToArgs
from train import train_ppo

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser = add_training_args(parser)
    # args = parser.parse_args()
    #
    # # args.exp_name = "test"
    # # args.env_name = "cartpole"
    # # args.n_envs = 2
    # # args.param_name = "cartpole"
    # # args.num_timesteps = 1000000
    # # args.num_checkpoints = 1
    # # args.seed = 6033
    # # args.use_wandb = False
    # # args.use_valid_env = False
    # # args.render = True
    #
    # args.env_name = "mountain_car"
    # args.start_level = 0
    # args.num_levels = 500
    # args.distribution_mode = "easy"
    # args.param_name = "mlpmodel"
    # args.device = "cpu"
    # args.gpu_device = 0
    # args.num_timesteps = 200000000
    # args.seed = 6033
    # args.log_level = 40
    # args.num_checkpoints = 1
    # args.gamma = 0.95
    # args.entropy_coef = 0.02
    # args.n_envs = 512
    # args.wandb_name = 8964
    # args.wandb_tags = ["discrete", "gravity"]
    # args.sparsity_coef = 0.0
    # args.random_percent = 0
    # args.key_penalty = 0
    # args.step_penalty = 0
    # args.rand_region = 0
    # args.num_threads = 8
    # # args.no-detect_nan = True
    # args.use_valid_env = True
    # args.normalize_rew = True
    # # args.no-render =
    # args.paint_vel_info = True
    # args.reduce_duplicate_actions = True
    # args.use_wandb = True
    # args.real_procgen = True
    # # args.no-mirror_env =
    logdir = "logs/train/acrobot/test/2024-04-29__18-42-26__seed_40"
    logdir = "logs/train/cartpole/2024-07-11__04-48-25__seed_6033"
    # model_file = latest_model_path(logdir)
    cfg = get_config(logdir)
    # hp_file = os.path.join(logdir, "hyperparameters.npy")
    # if os.path.exists(hp_file):
    #     hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    # cfg["model_file"] = model_file
    # cfg["device"] = "cpu"

    # May need to change this:
    # exclusions = ["algo", "epoch", "lmbda", "grad_clip_norm", "eps_clip", "value_coef", "normalize_adv", "use_gae",
    #               "architecture", "recurrent",
    #               "no-recurrent", "depth", "latent_size", "mid_weight"]
    # cfg = {k: v for k, v in cfg.items() if k not in exclusions}

    cfg["num_timesteps"] = int(2e8)

    cfg["learning_rate"] /= 2
    cfg["t_learning_rate"] /= 2

    args = DictToArgs(cfg)

    train_ppo(args)
