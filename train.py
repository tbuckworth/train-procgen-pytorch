from boxworld.create_box_world import create_box_world_env_pre_vec
from common.env.procgen_wrappers import VecExtractDictObs, VecNormalize, MirrorFrame, TransposeFrame, ScaledFloatFrame, \
    EncoderWrapper, create_rendered_env
from common.logger import Logger
from common.storage import Storage
from common.model import get_trained_vqvqae
from common import set_global_seeds, set_global_log_levels

import os, time, argparse
from procgen import ProcgenEnv
import random
import torch
import numpy as np

from helper import get_hyperparams, initialize_model, get_in_channels, add_training_args

try:
    import wandb
except ImportError:
    pass


def train_ppo(args):
    # exp_name, env_name, val_env_name, start_level, start_level_val, num_levels, distribution_mode, param_name, gpu_device, num_timesteps, seed, log_level, num_checkpoints, alpha, key, value, device, n_envs, env, listdir, logdir, d, filename, args
    exp_name = args.exp_name
    env_name = args.env_name
    val_env_name = args.val_env_name if args.val_env_name else args.env_name
    start_level = args.start_level
    start_level_val = random.randint(500, 9999)
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    gpu_device = args.gpu_device
    num_timesteps = int(args.num_timesteps)
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints
    set_global_seeds(seed)
    set_global_log_levels(log_level)
    if args.start_level == start_level_val:
        raise ValueError("Seeds for training and validation envs are equal.")
    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    hyperparameters = get_hyperparams(param_name)
    if args.mut_info_alpha is not None:
        alpha = args.mut_info_alpha
        ent_coef = hyperparameters["entropy_coef"]
        hyperparameters["entropy_coef"] = ent_coef * alpha
        hyperparameters["x_entropy_coef"] = ent_coef * (1 - alpha)

    for var_name in ["n_envs", "n_steps", "n_minibatch", "levels", "n_impala_blocks"]:
        if args.__dict__[var_name] is not None:
            hyperparameters[var_name] = args.__dict__[var_name]

    wandb_name = args.wandb_name
    if args.wandb_name is None:
        wandb_name = np.random.randint(1e5)
    for key, value in hyperparameters.items():
        print(key, ':', value)
    ############
    ## DEVICE ##
    ############
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise NotImplementedError(f"'device' should be one of 'cpu','gpu', not {args.device}")
    # For debugging nans:
    if args.detect_nan:
        torch.autograd.set_detect_anomaly(True)
    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZING ENVIRONMENTS...')
    # If Windows:
    if os.name == "nt":
        hyperparameters["n_envs"] = 2
        hyperparameters["use_wandb"] = False
        device = torch.device("cpu")
    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 256)
    max_steps = hyperparameters.get("max_steps", 10 ** 3)

    def create_venv(args, hyperparameters, is_valid=False):
        if args.real_procgen:
            venv = ProcgenEnv(num_envs=n_envs,
                              env_name=val_env_name if is_valid else env_name,
                              num_levels=0 if is_valid else args.num_levels,
                              start_level=start_level_val if is_valid else args.start_level,
                              paint_vel_info=hyperparameters.get("paint_vel_info", True),
                              distribution_mode=args.distribution_mode,
                              num_threads=args.num_threads,
                              )
        else:
            venv = ProcgenEnv(num_envs=n_envs,
                              env_name=val_env_name if is_valid else env_name,
                              num_levels=0 if is_valid else args.num_levels,
                              start_level=start_level_val if is_valid else args.start_level,
                              distribution_mode=args.distribution_mode,
                              num_threads=args.num_threads,
                              random_percent=args.random_percent,
                              step_penalty=args.step_penalty,
                              key_penalty=args.key_penalty,
                              rand_region=args.rand_region,
                              )

        venv = VecExtractDictObs(venv, "rgb")

        normalize_rew = hyperparameters.get('normalize_rew', True)
        mirror_some = hyperparameters.get('mirror_env', False)
        if normalize_rew:
            venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
            # the img frames
        if mirror_some:
            venv = MirrorFrame(venv)
        venv = TransposeFrame(venv)
        venv = ScaledFloatFrame(venv)
        if hyperparameters.get('architecture', "impala") == "vqmha":
            in_channels = get_in_channels(venv)
            vqvae = get_trained_vqvqae(in_channels, hyperparameters, device)
            venv = EncoderWrapper(venv, vqvae)
        return venv

    def create_bw_env(args, hyperparameters, is_valid=False):
        # n, goal_length, num_distractor, distractor_length, max_steps = 10 ** 6, collect_key = True, world = None, render_mode = None, seed = None
        env_args = {"n_envs": n_envs,
                    "n": hyperparameters.get('grid_size', 12),
                    "goal_length": hyperparameters.get('goal_length', 5),
                    "num_distractor": hyperparameters.get('num_distractor', 0),
                    "distractor_length": hyperparameters.get('distractor_length', 0),
                    "max_steps": max_steps,
                    "n_levels": num_levels,
                    "seed": args.seed,
                    }
        normalize_rew = hyperparameters.get('normalize_rew', True)
        if is_valid:
            env_args["n"] = hyperparameters.get('grid_size_v', 12)
            env_args["goal_length"] = hyperparameters.get('goal_length_v', 5)
            env_args["num_distractor"] = hyperparameters.get('num_distractor_v', 0)
            env_args["distractor_length"] = hyperparameters.get('distractor_length_v', 0)
            env_args["seed"] = args.seed + np.random.randint(1e6, 1e7) if env_args["n_levels"] == 0 else env_args[
                                                                                                             "n_levels"] + 1
            env_args["n_levels"] = 0
        return create_box_world_env_pre_vec(env_args, render=False, normalize_rew=normalize_rew)

    if args.env_name == "boxworld":
        create_venv = create_bw_env
    elif args.render:
        create_venv = create_rendered_env
    env = create_venv(args, hyperparameters)
    env_valid = create_venv(args, hyperparameters, is_valid=True) if args.use_valid_env else None

    ############
    ## LOGGER ##
    ############
    def listdir(path):
        return [os.path.join(path, d) for d in os.listdir(path)]

    def get_latest_model(model_dir):
        """given model_dir with files named model_n.pth where n is an integer,
        return the filename with largest n"""
        steps = [int(filename[6:-4]) for filename in os.listdir(model_dir) if filename.startswith("model_")]
        return list(os.listdir(model_dir))[np.argmax(steps)]

    print('INITIALIZING LOGGER...')
    logdir = os.path.join('logs', 'train', env_name, exp_name)
    if args.model_file == "auto":  # try to figure out which file to load
        logdirs_with_model = [d for d in listdir(logdir) if any(['model' in filename for filename in os.listdir(d)])]
        if len(logdirs_with_model) > 1:
            raise ValueError("Received args.model_file = 'auto', but there are multiple experiments"
                             f" with saved models under experiment_name {exp_name}.")
        elif len(logdirs_with_model) == 0:
            raise ValueError("Received args.model_file = 'auto', but there are"
                             f" no saved models under experiment_name {exp_name}.")
        model_dir = logdirs_with_model[0]
        args.model_file = os.path.join(model_dir, get_latest_model(model_dir))
        logdir = model_dir  # reuse logdir
    else:
        run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{seed}'
        logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    # write hyperparameters to file
    np.save(os.path.join(logdir, "hyperparameters.npy"), hyperparameters)
    print(f'Logging to {logdir}')

    cfg = vars(args)
    cfg.update(hyperparameters)
    np.save(os.path.join(logdir, "config.npy"), cfg)

    if args.use_wandb:
        wandb.login(key="cfc00eee102a1e9647b244a40066bfc5f1a96610")
        name = f"{hyperparameters['architecture']}-{wandb_name}"
        wb_resume = "allow" if args.model_file is None else "must"
        if env_name == "boxworld":
            project = "Box-World"
        elif exp_name == "coinrun-hparams":
            project = "Hparams Coinrun"
        elif env_name == "coinrun":
            project = "Coinrun VQMHA"
        else:
            project = env_name
        wandb.init(project=project, config=cfg, sync_tensorboard=True,
                   tags=args.wandb_tags, resume=wb_resume, name=name)
    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    model, observation_shape, policy = initialize_model(device, env, hyperparameters)
    logger = Logger(n_envs, logdir, use_wandb=args.use_wandb, has_vq=policy.has_vq)
    logger.max_steps = max_steps
    #############
    ## STORAGE ##
    #############
    print('INITIALIZING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    storage_valid = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device) if args.use_valid_env else None
    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError

    def nan_hook(self, inp, output):
        if not isinstance(output, tuple):
            outputs = [output]
        else:
            outputs = output

        for i, out in enumerate(outputs):
            if out is not None:
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                                       out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


    if args.detect_nan:
        for submodule in model.modules():
            submodule.register_forward_hook(nan_hook)


    agent = AGENT(env, policy, logger, storage, device,
                  num_checkpoints,
                  env_valid=env_valid,
                  storage_valid=storage_valid,
                  **hyperparameters)
    if args.model_file is not None:
        print("Loading agent from %s" % args.model_file)
        checkpoint = torch.load(args.model_file, map_location=device)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    # args.exp_name = "test"
    # args.env_name = "coinrun"
    # args.num_levels = 500
    # args.distribution_mode = "hard"
    # args.start_level = 0
    # args.param_name = "hard-500-impalafsqmha"
    # args.num_timesteps = 1000000
    # args.num_checkpoints = 1
    # args.seed = 6033
    # args.mirror_env = True
    # args.use_wandb = False
    # args.use_valid_env = False
    # args.render = True
    # args.paint_vel_info = True
    # #
    # # # args.model_file = 'C:/Users/titus/PycharmProjects/train-procgen-pytorch/logs/train/coinrun/coinrun/2024-02-08__15-31-22__seed_6033/model_80019456.pth'
    # # args.model_file = 'C:/Users/titus/PycharmProjects/train-procgen-pytorch/logs/train/coinrun/coinrun/2024-02-11__08-41-38__seed_6033/model_50003968.pth'
    # args.model_file = "C:/Users/titus/PycharmProjects/train-procgen-pytorch/logs/train/coinrun/coinrun/2024-02-12__09-20-18__seed_6033/model_100007936.pth"
    train_ppo(args)
