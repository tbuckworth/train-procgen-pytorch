import argparse
import random

from train import train_ppo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun', help='environment ID')
    parser.add_argument('--val_env_name', type=str, default=None, help='optional validation environment ID')
    parser.add_argument('--start_level', type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(500), help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='cpu', required=False, help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--num_timesteps', type=int, default=int(25000000), help='number of training timesteps')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1), help='number of checkpoints to store')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--real_procgen', action="store_true", default=True)
    parser.add_argument('--mirror_env', action="store_true", default=False)
    parser.add_argument('--mut_info_alpha', type=float, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--n_minibatch', type=int, default=None)
    parser.add_argument('--detect_nan', action="store_true", default=False)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--paint_vel_info', action="store_true", default=True)

    parser.add_argument('--minibatches', type=int, nargs='+')

    parser.add_argument('--wandb_tags', type=str, nargs='+')

    parser.add_argument('--random_percent', type=int, default=0,
                        help='COINRUN: percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--key_penalty', type=int, default=0,
                        help='HEIST_AISC: Penalty for picking up keys (divided by 10)')
    parser.add_argument('--step_penalty', type=int, default=0,
                        help='HEIST_AISC: Time penalty per step (divided by 1000)')
    parser.add_argument('--rand_region', type=int, default=0,
                        help='MAZE: size of region (in upper left corner) in which goal is sampled.')

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)

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

