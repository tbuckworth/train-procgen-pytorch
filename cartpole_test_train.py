import argparse
import re

from helper_local import add_training_args, DictToArgs
from train import train_ppo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    args.exp_name = "test"
    args.env_name = "cartpole"
    # args.n_envs = 2
    args.param_name = "graph-cartpole"
    args.num_timesteps = int(2e8)
    args.num_checkpoints = 1
    args.seed = 6033
    args.use_wandb = True
    args.use_valid_env = True
    # args.render = True
    train_ppo(args)
    exit(0)

    # args.env_name = "cartpole"
    # args.start_level = 0
    # args.num_levels = 500
    # args.distribution_mode = "easy"
    # args.param_name = "graph-transition"
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

    arg_str = "--env_name cartpole-swing --start_level 0 --num_levels 500 --distribution_mode easy --param_name graph-transition --device gpu --gpu_device 0 --num_timesteps 2000000 --seed 6033 --log_level 40 --num_checkpoints 10 --gamma 0.95 --learning_rate 0.0005 --n_envs 64 --n_steps 256 --wandb_name 9602 --wandb_tags graph-transition large-output-dim added-cont --sparsity_coef 0.0 --random_percent 0 --key_penalty 0 --step_penalty 0 --rand_region 0 --num_threads 8 --no-detect_nan --use_valid_env --normalize_rew --no-render --paint_vel_info --reduce_duplicate_actions --use_wandb --real_procgen --no-mirror_env --done_coef 10.0 --val_epochs 8 --dyn_epochs 5 --t_learning_rate 0.001 --n_rollouts 3 --temperature 100 --use_gae --rew_coef 1 --clip_value --output_dim 256  2>&1 | tee /vol/bitbucket/${USER}/train-procgen-pytorch/scripts/train_9602.out"
    arg_str = re.sub(r"\s*2>&1.*", "", arg_str)
    al = re.split("--", arg_str)
    temp_dict = {}
    for a in al:
        if len(a) > 0:
            strs = re.split(r"\s", a)
            strs = [s for s in strs if not s == '']
            if len(strs) == 1:
                arg_val = True
            elif len(strs) == 2:
                arg_val = strs[-1]
            else:
                arg_val = strs[1:]
            temp_dict[strs[0]] = arg_val
    # args = DictToArgs(temp_dict)
    for var_name, val in temp_dict.items():
        args.__dict__[var_name] = val
    train_ppo(args)
