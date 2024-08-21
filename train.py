from common.logger import Logger
from common.storage import Storage, BasicStorage
from common import set_global_seeds, set_global_log_levels

import os, time, argparse
import torch
import numpy as np

from helper_local import get_hyperparams, initialize_model, add_training_args, wandb_login, get_project, \
    get_agent_constructor
from common.env.env_constructor import get_env_constructor

try:
    import wandb
except ImportError:
    pass


def train_ppo(args):
    # exp_name, env_name, val_env_name, start_level, start_level_val, num_levels, distribution_mode, param_name, gpu_device, num_timesteps, seed, log_level, num_checkpoints, alpha, key, value, device, n_envs, env, listdir, logdir, d, filename, args
    exp_name = args.exp_name
    env_name = args.env_name
    param_name = args.param_name
    gpu_device = args.gpu_device
    num_timesteps = int(args.num_timesteps)
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints
    set_global_seeds(seed)
    set_global_log_levels(log_level)

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

    for var_name in [
        "n_envs",
        "n_steps",
        "n_minibatch",
        "mini_batch_size",
        "levels",
        "n_impala_blocks",
        "eps_clip",
        "increasing_lr",
        "sparsity_coef",
        "normalize_rew",
        "gamma",
        "lmbda",
        "learning_rate",
        "t_learning_rate",
        "dr_learning_rate",
        "entropy_coef",
        "fs_coef",
        "output_dim",
        "n_epochs",
        "n_rollouts",
        "temperature",
        "use_gae",
        "clip_value",
        "done_coef",
        "dyn_epochs",
        "val_epochs",
        "dr_epochs",
        "rew_coef",
        "anneal_temp",
        "epoch",
        "value_coef",
        "t_coef",
        "num_timesteps",
    ]: #TODO: should this just be all the keys, rather than specify them? why not?
        if var_name in args.__dict__.keys() and args.__dict__[var_name] is not None:
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

    create_venv = get_env_constructor(args.env_name)

    env = create_venv(args, hyperparameters)
    env_valid = create_venv(args, hyperparameters, is_valid=True) if args.use_valid_env else None

    try:
        env_params = env.get_params()
        env_params_v = env_valid.get_params(suffix="_v")
        upload_env_params = True
    except AttributeError:
        env_params = {}
        env_params_v = {}
        upload_env_params = False

    ############
    ## LOGGER ##
    ############

    print('INITIALIZING LOGGER...')
    logdir = create_logdir_train(args.model_file, env_name, exp_name, seed)
    # write hyperparameters to file
    np.save(os.path.join(logdir, "hyperparameters.npy"), hyperparameters)
    print(f'Logging to {logdir}')

    cfg = vars(args)
    cfg.update(hyperparameters)
    np.save(os.path.join(logdir, "config.npy"), cfg)

    if args.use_wandb:
        if upload_env_params:
            cfg.update(env_params)
            cfg.update(env_params_v)
        cfg["logdir"] = logdir
        wandb_login()
        name = f"{hyperparameters['architecture']}-{wandb_name}"
        wb_resume = "allow"  # if args.model_file is None else "must"
        project = get_project(env_name, exp_name)
        if args.wandb_group is not None:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name, group=args.wandb_group)
        else:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name)
    ###########
    ## MODEL ##
    ###########
    algo = hyperparameters.get('algo', 'ppo')
    model_based = algo in ['ppo-model', 'graph-agent']
    double_graph = algo in ['double-graph-agent']

    print('INTIALIZING MODEL...')
    model, observation_shape, policy = initialize_model(device, env, hyperparameters)
    logger = Logger(n_envs, logdir, use_wandb=args.use_wandb, has_vq=policy.has_vq,
                    transition_model=model_based, double_graph=double_graph)
    logger.max_steps = max_steps
    #############
    ## STORAGE ##
    #############
    print('INITIALIZING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    storage_valid = Storage(observation_shape, hidden_state_dim, n_steps, n_envs,
                            device) if args.use_valid_env else None
    if model_based or double_graph:
        storage = BasicStorage(observation_shape, n_steps, n_envs, device)
        storage_valid = BasicStorage(observation_shape, n_steps, n_envs,
                                     device) if args.use_valid_env else None

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    AGENT = get_agent_constructor(algo)

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
        agent.v_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)
    wandb.finish()
    return logdir


def create_logdir_train(model_file, env_name, exp_name, seed):
    def listdir(path):
        return [os.path.join(path, d) for d in os.listdir(path)]

    def get_latest_model(model_dir):
        """given model_dir with files named model_n.pth where n is an integer,
        return the filename with largest n"""
        steps = [int(filename[6:-4]) for filename in os.listdir(model_dir) if filename.startswith("model_")]
        return list(os.listdir(model_dir))[np.argmax(steps)]

    logdir = os.path.join('logs', 'train', env_name, exp_name)
    if model_file == "auto":  # try to figure out which file to load
        logdirs_with_model = [d for d in listdir(logdir) if any(['model' in filename for filename in os.listdir(d)])]
        if len(logdirs_with_model) > 1:
            raise ValueError("Received args.model_file = 'auto', but there are multiple experiments"
                             f" with saved models under experiment_name {exp_name}.")
        elif len(logdirs_with_model) == 0:
            raise ValueError("Received args.model_file = 'auto', but there are"
                             f" no saved models under experiment_name {exp_name}.")
        model_dir = logdirs_with_model[0]
        model_file = os.path.join(model_dir, get_latest_model(model_dir))
        logdir = model_dir  # reuse logdir
    else:
        run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{seed}'
        logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    return logdir


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
