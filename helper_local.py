import os
import random
import re
import subprocess
import time
import json
from collections import deque

import gym
import gymnasium
import numpy as np
import pandas as pd

import wandb
import yaml
import platform
from matplotlib import pyplot as plt

from common.model import NatureModel, ImpalaModel, MHAModel, ImpalaVQModel, ImpalaVQMHAModel, ImpalaFSQModel, ribMHA, \
    ImpalaFSQMHAModel, RibFSQMHAModel, MLPModel, TransformoBot, GraphTransitionModel, ImpalaCNN, \
    GraphValueModel, GraphActorCritic, GraphActorCriticEQL
from common.policy import TransitionPolicy, PixelTransPolicy, GraphTransitionPolicy, \
    DoubleTransitionPolicy, GraphPolicy, GoalSeekerPolicy
from moviepy.editor import ImageSequenceClip

from common.storage import Storage, BasicStorage, IPLStorage, GoalSeekerStorage, SAEStorage

GLOBAL_DIR = "/vol/bitbucket/tfb115/train-procgen-pytorch/"
OS_IS = "Linux"


def is_wsl(v: str = platform.uname().release) -> int:
    """
    detects if Python is running in WSL
    """
    if v.endswith("-Microsoft"):
        return 1
    elif v.endswith("microsoft-standard-WSL2"):
        return 2
    return 0


try:
    if os.name == "nt":
        GLOBAL_DIR = "C:/Users/titus/PycharmProjects/train-procgen-pytorch/"
        OS_IS = "Windows"
    elif os.getlogin() == "titus":
        GLOBAL_DIR = "/home/titus/PycharmProjects/train-procgen-pytorch/"
        OS_IS = "Linux"
    if is_wsl() == 2:
        GLOBAL_DIR = "/mnt/c/Users/titus/PycharmProjects/train-procgen-pytorch/"
        OS_IS = "WSL"
except Exception as e:
    pass

import torch


def match(a, b):
    a = a.tolist()
    b = b.tolist()
    return np.array([b.index(x) for x in a if x in b])


def match(a, b, dtype=np.int32):
    if len(a.shape) > 1:
        return np.array([match(x, b) for x in a], dtype=dtype)
    a = a.tolist()
    b = b.tolist()
    return np.array([b.index(x) for x in a if x in b], dtype=dtype)


# def match_big(a, b, dtype=np.int32):
#     sorted = np.argsort(a)
#     r = np.searchsorted(a, b, side='right', sorter=sorted)
#     l = np.searchsorted(a, b, side='left', sorter=sorted)
#     for b, e in zip(l, r):
#         inds = sorted[b:e]
#         print(inds)


def save_gif(frames, filename="test.gif", fps=20):
    frames = np.array(frames.transpose(0, 2, 3, 1) * 255, dtype=np.uint8)
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(filename, fps=fps)


def get_path(folder, file):
    return os.path.join(folder, file)
    # return os.path.join(GLOBAL_DIR, folder, file)


def print_values_actions(action_names, pi, value, i="", rewards=None):
    ap = np.squeeze(pi[0])
    df = pd.DataFrame({"variables": action_names, "values": ap})
    df2 = df.pivot_table(values="values", index="variables", aggfunc="sum")
    a_names = df2.index.to_numpy()
    max_len = max([len(x) for x in a_names])
    a_names = np.array([x + ''.join([' ' for _ in range(max_len - len(x))]) for x in a_names])
    action_probs = np.asarray(np.round(np.squeeze(df2.values) * 100, 0), dtype=np.int32)
    idx = np.argsort(action_probs)[::-1][:5]
    top_actions = '\t'.join([f"{x[0]}: {x[1]}" for x in zip(action_probs[idx], a_names[idx])])
    rew_str = ""
    if rewards is not None:
        rew_str = f"{np.mean(rewards):.2f}\t"
    out_str = f"{np.squeeze(value[0]):.2f}\t{rew_str}{top_actions}"
    if i != "":
        print(f"{i}:\t{out_str}")
    else:
        print(out_str)


def map_actions_to_values(actions):
    eighth = np.pi / 4
    mapping = {
        'move up': 0,
        'UP': 0,
        'RIGHT_UP': eighth,
        'move right': 2 * eighth,
        'RIGHT': 2 * eighth,
        'RIGHT_DOWN': 3 * eighth,
        'move down': 4 * eighth,
        'DOWN': 4 * eighth,
        'LEFT_DOWN': 5 * eighth,
        'move left': 6 * eighth,
        'LEFT': 6 * eighth,
        'LEFT_UP': 7 * eighth,
        '': -1,
        'acc left': -1,
        'none': 0,
        'acc right': 1,
        'push left': 0,
        'push right': 1,
        'neg torque': -1,
        'no torque': 0,
        'pos torque': 1,
    }

    return np.array([mapping[key] for key in actions])


def print_action_entropy(action_names, pi):
    val_names = [f"env{i}" for i in range(len(pi))]
    df = pd.DataFrame({**{"variables": action_names}, **{f"env{i}": x for i, x in enumerate(pi)}})
    df2 = df.pivot_table(values=val_names, index="variables", aggfunc="sum")
    scaled_entropy = -(df2[val_names] * np.log(df2[val_names])).sum(0) / np.log(len(df2))
    df2.loc["Entropy(%)"] = scaled_entropy
    df2[val_names] = np.asarray(np.round(np.squeeze(df2[val_names]) * 100, 0), dtype=np.int32)
    print(df2)


def get_actions_from_all(env):
    try:
        return get_action_names(env)
    except NotImplementedError:
        pass
    try:
        return np.array(list(get_actions(env).values()))
    except NotImplementedError:
        raise NotImplementedError("No combos or actions found")


def get_combos(env):
    if hasattr(env, "combos"):
        return env.combos
    if hasattr(env, "env"):
        return get_combos(env.env)
    raise NotImplementedError("No combos found in env")


def add_encoder_to_env(env, encoder):
    if hasattr(env, "encoder"):
        env.encoder = encoder
        return
    if hasattr(env, "env"):
        add_encoder_to_env(env.env, encoder)
        return
    if hasattr(env, "venv"):
        add_encoder_to_env(env.venv, encoder)
        return
    raise NotImplementedError("No env wrapper in the onion has encoder parameter")


def get_actions(env):
    try:
        return env.get_action_lookup()
    except Exception:
        pass
    if hasattr(env, "unwrapped"):
        return env.unwrapped.get_action_lookup()
    if hasattr(env, "env"):
        return get_actions(env.env)
    if hasattr(env, "venv"):
        return get_actions(env.venv)
    raise NotImplementedError("No env wrapper in the onion has get_action_lookup method")


def get_action_names(env):
    action_names = np.array(
        [x[0] + "_" + x[1] if len(x) == 2 else (x[0] if len(x) == 1 else "") for x in get_combos(env)])
    x = np.array(['D', 'A', 'W', 'S', 'Q', 'E'])
    y = np.array(['RIGHT', 'LEFT', 'UP', 'DOWN', 'LEFT_UP', 'RIGHT_UP'])
    action_names[match(x, action_names)] = y[match(action_names, x)]
    return action_names


def get_hyperparams(param_name):
    with open(os.path.join(GLOBAL_DIR, 'hyperparams/procgen/config.yml'), 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    return hyperparameters


def initialize_model(device, env, hyperparameters, in_channels=None):
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    cust_policy = hyperparameters.get('cust_policy', None)
    if in_channels is None:
        in_channels = observation_shape[0]
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
    elif isinstance(action_space, gymnasium.spaces.Discrete):
        action_size = action_space.n
    elif isinstance(action_space, gymnasium.spaces.Box) or isinstance(action_space, gym.spaces.Box):
        action_size = action_space.shape[-1] * 2
    else:
        raise NotImplementedError
    has_vq = False
    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        output_dim = hyperparameters.get("output_dim", 256)
        latent_dim = hyperparameters.get("latent_dim", 32)
        model = ImpalaModel(in_channels=in_channels, output_dim=output_dim, latent_dim=latent_dim)
    elif architecture == 'vqmha':
        has_vq = False
        mha_layers = hyperparameters.get("mha_layers", 1)
        n_latents = hyperparameters.get("n_latents", 1)
        latent_dim = hyperparameters.get("latent_dim", 1)
        output_dim = hyperparameters.get("output_dim", 256)
        model = MHAModel(n_latents, latent_dim, mha_layers, output_dim, device)
    elif architecture == 'impalavq':
        has_vq = True
        model = ImpalaVQModel(in_channels=in_channels)
    elif architecture == 'impalavqmha':
        has_vq = True
        mha_layers = hyperparameters.get("mha_layers", 1)
        use_vq = hyperparameters.get("use_vq", True)
        model = ImpalaVQMHAModel(in_channels=in_channels, mha_layers=mha_layers, device=device, use_vq=use_vq,
                                 obs_shape=observation_shape)
    elif architecture == 'impalafsq':
        model = ImpalaFSQModel(in_channels, device)
    elif architecture == 'impalafsqmha':
        mha_layers = hyperparameters.get("mha_layers", 2)
        reduce = hyperparameters.get('pool_direction', 'feature_wise')
        levels = hyperparameters.get('levels')
        n_impala_blocks = hyperparameters.get("n_impala_blocks", 3)
        model = ImpalaFSQMHAModel(in_channels, mha_layers, device, observation_shape, reduce,
                                  n_impala_blocks=n_impala_blocks, levels=levels)
    elif architecture == 'impalafsqitn':
        mha_layers = hyperparameters.get("mha_layers", 2)
        reduce = hyperparameters.get('pool_direction', 'feature_wise')
        levels = hyperparameters.get('levels')
        n_impala_blocks = hyperparameters.get("n_impala_blocks", 3)
        use_intention = hyperparameters.get("use_intention", True)
        model = ImpalaFSQMHAModel(in_channels, mha_layers, device, observation_shape, reduce,
                                  n_impala_blocks=n_impala_blocks, levels=levels, use_intention=use_intention)
    elif architecture == 'impalamha':
        mha_layers = hyperparameters.get("mha_layers", 2)
        reduce = hyperparameters.get('pool_direction', 'feature_wise')
        levels = hyperparameters.get('levels')
        n_impala_blocks = hyperparameters.get("n_impala_blocks", 3)
        latent_override = hyperparameters.get("latent_dim", 16)
        model = ImpalaFSQMHAModel(in_channels, mha_layers, device, observation_shape, reduce,
                                  n_impala_blocks=n_impala_blocks, levels=levels, no_quant=True,
                                  latent_override=latent_override)
    elif architecture == 'impalaitn':
        mha_layers = hyperparameters.get("mha_layers", 2)
        reduce = hyperparameters.get('pool_direction', 'feature_wise')
        levels = hyperparameters.get('levels')
        n_impala_blocks = hyperparameters.get("n_impala_blocks", 3)
        use_intention = hyperparameters.get("use_intention", True)
        latent_override = hyperparameters.get("latent_dim", 16)
        model = ImpalaFSQMHAModel(in_channels, mha_layers, device, observation_shape, reduce,
                                  n_impala_blocks=n_impala_blocks, levels=levels, use_intention=use_intention,
                                  no_quant=True, latent_override=latent_override)
    elif architecture == 'ribmha':
        model = ribMHA(in_channels, device, observation_shape)
    elif architecture == 'ribfsqmha':
        mha_layers = hyperparameters.get("mha_layers", 2)
        reduce = hyperparameters.get('pool_direction', 'feature_wise')
        levels = hyperparameters.get('levels')
        model = RibFSQMHAModel(in_channels, mha_layers, device, observation_shape, reduce, levels=levels)
    elif architecture == 'mlpmodel':
        depth = hyperparameters.get("depth", 4)
        mid_weight = hyperparameters.get("mid_weight", 64)
        latent_size = hyperparameters.get("latent_size", 256)
        model = MLPModel(in_channels, depth, mid_weight, latent_size)

    elif architecture == 'transformobot':
        n_layers = hyperparameters.get("n_layers", 2)
        # n_heads = hyperparameters.get("n_heads", 1)
        model = TransformoBot(in_channels, n_layers)
    elif architecture == "graph-transition":

        depth = hyperparameters.get("depth", 4)
        mid_weight = hyperparameters.get("mid_weight", 64)
        output_dim = hyperparameters.get("output_dim", 256)
        model = MLPModel(in_channels, depth, mid_weight, output_dim)

        gamma = hyperparameters.get("gamma", 0.99)
        temperature = hyperparameters.get("temperature", 1)
        n_rollouts = hyperparameters.get("n_rollouts", 3)
        depth = hyperparameters.get("depth", 4)
        mid_weight = hyperparameters.get("mid_weight", 64)
        latent_size = hyperparameters.get("latent_size", 256)

        transition_model = GraphTransitionModel(in_channels, depth, mid_weight, latent_size, device)
        action_size = action_space.n

        policy = TransitionPolicy(model, transition_model, action_size, n_rollouts, temperature, gamma)
        policy.to(device)
        policy.device = device
        return model, observation_shape, policy

    elif architecture == "pixel-graph-transition":
        mid_weight = hyperparameters.get("encoder_mid_channels", 64)
        latent_dim = hyperparameters.get("encoder_latent_dim", 3) + 2
        n_impala_blocks = hyperparameters.get("n_impala_blocks", 3)
        encoder = ImpalaCNN(in_channels, mid_weight, latent_dim, n_impala_blocks)

        n_latents = encoder.get_n_latents(env.reset().shape)
        latent_channels = n_latents * (latent_dim - 2)

        hp = hyperparameters.copy()
        hp["architecture"] = "graph-transition"

        _, _, sub_policy = initialize_model(device, env, hp, latent_channels)

        policy = PixelTransPolicy(encoder, sub_policy)
        return encoder, observation_shape, policy
    elif architecture == "full-graph-transition":
        gamma = hyperparameters.get("gamma", 0.99)
        temperature = hyperparameters.get("temperature", 1)
        n_rollouts = hyperparameters.get("n_rollouts", 3)
        depth = hyperparameters.get("depth", 4)
        mid_weight = hyperparameters.get("mid_weight", 64)
        latent_size = hyperparameters.get("latent_size", 256)
        # TODO: change in_channels?
        transition_model = GraphTransitionModel(in_channels, depth, mid_weight, latent_size, device)
        action_size = action_space.n

        policy = GraphTransitionPolicy(transition_model, action_size, n_rollouts, temperature, gamma)
        policy.to(device)
        policy.device = device
        return transition_model, observation_shape, policy
    elif architecture == 'double-graph':

        gamma = hyperparameters.get("gamma", 0.99)
        temperature = hyperparameters.get("temperature", 1)
        n_rollouts = hyperparameters.get("n_rollouts", 3)
        depth = hyperparameters.get("depth", 4)
        mid_weight = hyperparameters.get("mid_weight", 64)
        latent_size = hyperparameters.get("latent_size", 256)

        value_model = GraphValueModel(in_channels, depth, mid_weight, latent_size, device)
        transition_model = GraphTransitionModel(in_channels, depth, mid_weight, latent_size, device)
        action_size = action_space.n
        policy = DoubleTransitionPolicy(value_model, transition_model, action_size, n_rollouts,
                                        temperature, gamma, env.done_func, env.rew_func)
        policy.to(device)
        policy.device = device
        return value_model, observation_shape, policy
    elif architecture == 'graph':
        depth = hyperparameters.get("depth", 4)
        mid_weight = hyperparameters.get("mid_weight", 64)
        latent_size = hyperparameters.get("latent_size", 1)
        cont_act = hyperparameters.get("continuous", False)
        simple_scaling = hyperparameters.get("simple_scaling", True)
        if cont_act:
            action_size = action_space.shape[-1]
        else:
            action_size = action_space.n
        graph = GraphActorCritic(in_channels, depth, mid_weight, latent_size,
                                 action_size, device, continuous_actions=cont_act)
        policy = GraphPolicy(graph, continuous_actions=cont_act, act_space=action_space, device=device,
                             simple_scaling=simple_scaling)
        policy.to(device)
        return graph, observation_shape, policy
    elif architecture == 'goal-seeker-mlp':
        cont_act = hyperparameters.get("continuous", False)
        if cont_act:
            action_size = action_space.shape[-1]
        else:
            action_size = action_space.n
        depth = hyperparameters.get("depth", 4)

        mid_weight = hyperparameters.get("mid_weight", 64)
        latent_size = hyperparameters.get("latent_size", 64)

        model = MLPModel(in_channels, depth, mid_weight, latent_size)

        policy = GoalSeekerPolicy(model, action_size)
        policy.to(device)
        return model, observation_shape, policy
    elif architecture == 'eql-graph':
        depth = hyperparameters.get("depth", 4)
        mid_weight = hyperparameters.get("mid_weight", 256)
        latent_size = hyperparameters.get("latent_size", 1)
        cont_act = hyperparameters.get("continuous", False)
        simple_scaling = hyperparameters.get("simple_scaling", True)
        eql_args = dict(
            target_ratio=hyperparameters["target_ratio"],
            spls=hyperparameters["spls"],
            constrain_scale=hyperparameters["constrain_scale"],
            l0_scale=hyperparameters["l0_scale"],
            bl0_scale=hyperparameters["bl0_scale"],
            target_temp=hyperparameters["target_temp"],
            warmup_epoch=hyperparameters["warmup_epoch"],
            # hard_epoch=0,#TODO: sort this out
            hard_epoch=0,  # hyperparameters["epochs"] * hyperparameters["hard_ratio"],
            sample_num=hyperparameters["sample_num"],
            hard_gum=hyperparameters["hard_gum"],
        )
        if cont_act:
            action_size = action_space.shape[-1]
        else:
            action_size = action_space.n
        graph = GraphActorCriticEQL(in_channels, eql_args, depth, mid_weight, latent_size, action_size, device,
                                    continuous_actions=cont_act)
        policy = GraphPolicy(graph, continuous_actions=cont_act, act_space=action_space, device=device,
                             simple_scaling=simple_scaling)
        policy.to(device)
        return graph, observation_shape, policy
    else:
        raise NotImplementedError(f"Architecture:{architecture} not found in helper.py")
    if cust_policy == "ICM":
        from common.policy import ICMPolicy as policy_cons
    else:
        from common.policy import CategoricalPolicy as policy_cons
    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    logsumexp_logits_is_v = hyperparameters.get('logsumexp_logits_is_v', False)

    policy = policy_cons(model, recurrent, action_size, has_vq, logsumexp_logits_is_v=logsumexp_logits_is_v)
    policy.to(device)
    policy.device = device
    return model, observation_shape, policy


def dict_to_html_table(name_deets):
    specs = {"metric": list(name_deets.keys()), "value": list(name_deets.values())}
    df2 = pd.DataFrame(specs)
    text = df2.to_html(index=False)
    return text


def create_name_from_dict(prefix, suffix, specifications, exclusions=None):
    if exclusions is None:
        exclusions = []
    if type(specifications) == str:
        outstr = "_" + specifications
    else:
        outstr = ""
        for key in specifications.keys():
            if key not in exclusions:
                outstr += f"_{key}_{specifications[key]}"
    if suffix == "":
        return f"{prefix}{outstr}"
    return f"{prefix}{outstr}.{suffix}"


def create_logdir(seed, folder, project, subfolder):
    logdir = os.path.join('logs', folder, project, subfolder)
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{seed}'
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    return logdir


def impala_latents(model, obs):
    x = model.block1(obs)
    x = model.block2(x)
    x = model.block3(x)
    return x


def plot_reconstructions(data, filename):
    train_reconstructions = data["train_reconstructions"]
    valid_reconstructions = data["valid_reconstructions"]
    train_batch = data["train_batch"]
    valid_batch = data["valid_batch"]

    def convert_batch_to_image_grid(image_batch, image_size=64):
        reshaped = (image_batch.reshape(4, 8, image_size, image_size, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(4 * image_size, 8 * image_size, 3))
        return reshaped

    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(2, 2, 1)
    ax.imshow(convert_batch_to_image_grid(train_batch),
              interpolation='nearest')
    ax.set_title('training data originals')
    plt.axis('off')
    ax = f.add_subplot(2, 2, 2)
    ax.imshow(convert_batch_to_image_grid(train_reconstructions),
              interpolation='nearest')
    ax.set_title('training data reconstructions')
    plt.axis('off')
    ax = f.add_subplot(2, 2, 3)
    ax.imshow(convert_batch_to_image_grid(valid_batch),
              interpolation='nearest')
    ax.set_title('validation data originals')
    plt.axis('off')
    ax = f.add_subplot(2, 2, 4)
    ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
              interpolation='nearest')
    ax.set_title('validation data reconstructions')
    plt.axis('off')
    plt.savefig(filename)


def last_folder(dir, n=1):
    files = [os.path.join(dir, x) for x in os.listdir(dir)]
    sl_files = {x: os.path.getmtime(x) for x in files}
    if n == 1:
        return max(sl_files, key=sl_files.get)
    sl_files = dict(sorted(sl_files.items(), key=lambda item: item[1]))
    return list(sl_files.keys())[-n]


def get_model_with_largest_checkpoint(folder):
    files = [os.path.join(folder, x) for x in os.listdir(folder)]
    search = lambda x: re.search(r"model_(\d*).pth", x)
    last_checkpoint = max([int(search(x).group(1)) for x in files if search(x)])
    return [x for x in files if re.search(f"model_{last_checkpoint}.pth", x)][0]


def get_latest_file_matching(pattern, n, folder=""):
    if folder == "":
        files = os.listdir()
    else:
        files = [os.path.join(folder, x) for x in os.listdir(folder)]
    sl_files = {x: os.path.getmtime(x) for x in files if re.search(pattern, x)}
    if n == 1:
        return max(sl_files, key=sl_files.get)
    sl_files = dict(sorted(sl_files.items(), key=lambda item: item[1]))
    return list(sl_files.keys())[-n]


def get_in_channels(venv):
    observation_space = venv.observation_space
    observation_shape = observation_space.shape
    in_channels = observation_shape[0]
    return in_channels


def add_training_args(parser):
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
    parser.add_argument('--mut_info_alpha', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--lmbda', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--t_learning_rate', type=float, default=None)
    parser.add_argument('--dr_learning_rate', type=float, default=None)
    parser.add_argument('--entropy_coef', type=float, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--n_minibatch', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--dyn_epochs', type=int, default=None)
    parser.add_argument('--val_epochs', type=int, default=None)
    parser.add_argument('--dr_epochs', type=int, default=None)

    parser.add_argument('--n_rollouts', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--done_coef', type=float, default=None)
    parser.add_argument('--rew_coef', type=float, default=None)

    parser.add_argument('--mini_batch_size', type=int, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs='+')
    parser.add_argument('--minibatches', type=int, nargs='+')
    parser.add_argument('--levels', type=int, nargs='+', default=None)
    parser.add_argument('--sparsity_coef', type=float, default=0.)
    parser.add_argument('--output_dim', type=int, default=256)
    parser.add_argument('--fs_coef', type=float, default=0.)

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

    parser.add_argument('--detect_nan', action="store_true")
    parser.add_argument('--use_valid_env', action="store_true")
    parser.add_argument('--normalize_rew', action="store_true")
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--paint_vel_info', action="store_true")
    parser.add_argument('--reduce_duplicate_actions', action="store_true")
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--real_procgen', action="store_true")
    parser.add_argument('--mirror_env', action="store_true")
    parser.add_argument('--use_gae', action="store_true")
    parser.add_argument('--clip_value', action="store_true")
    parser.add_argument('--anneal_temp', action="store_true")
    parser.add_argument('--use_greedy_env', action="store_true")
    parser.add_argument('--learned_gamma', action="store_true")

    parser.add_argument('--no-learned_gamma', dest='detect_nan', action="store_false")
    parser.add_argument('--no-use_greedy_env', dest='detect_nan', action="store_false")
    parser.add_argument('--no-detect_nan', dest='detect_nan', action="store_false")
    parser.add_argument('--no-use_valid_env', dest='use_valid_env', action="store_false")
    parser.add_argument('--no-normalize_rew', dest='normalize_rew', action="store_false")
    parser.add_argument('--no-render', dest='render', action="store_false")
    parser.add_argument('--no-paint_vel_info', dest='paint_vel_info', action="store_false")
    parser.add_argument('--no-reduce_duplicate_actions', dest='reduce_duplicate_actions', action="store_false")
    parser.add_argument('--no-use_wandb', dest='use_wandb', action="store_false")
    parser.add_argument('--no-real_procgen', dest='real_procgen', action="store_false")
    parser.add_argument('--no-mirror_env', dest='mirror_env', action="store_false")
    parser.add_argument('--no-use_gae', dest='use_gae', action="store_false")
    parser.add_argument('--no-clip_value', dest='clip_value', action="store_false")
    parser.add_argument('--no-anneal_temp', dest='anneal_temp', action="store_false")

    parser.set_defaults(detect_nan=False,
                        use_valid_env=True,
                        normalize_rew=True,
                        render=False,
                        paint_vel_info=True,
                        reduce_duplicate_actions=True,
                        use_wandb=False,
                        real_procgen=True,
                        mirror_env=False,
                        use_gae=True,
                        clip_value=True,
                        anneal_temp=False,
                        use_greedy_env=False,
                        learned_gamma=False,
                        )

    return parser


def coords_to_image(atn_coor, atn_size, image_size):
    # atn_coor, atn_size, image_size = arr[1], atn.shape[-1], observation.shape[-1]
    x = atn_size ** 0.5
    ratio = image_size / x
    r = (atn_coor // x) * ratio + ratio / 2
    c = (atn_coor % x) * ratio + ratio / 2
    return r, c


def get_config(logdir):
    return np.load(os.path.join(GLOBAL_DIR, logdir, "config.npy"), allow_pickle='TRUE').item()


def balanced_reward(done, info, performance_track):
    completes = np.array(info)[done]
    for info in completes:
        seed = info["prev_level_seed"]
        rew = info["env_reward"]
        if seed not in performance_track.keys():
            performance_track[seed] = deque(maxlen=10)
        performance_track[seed].append(rew)
    all_rewards = list(performance_track.values())
    true_average_reward = np.mean([rew for rew_list in all_rewards for rew in rew_list])
    return true_average_reward


def append_to_csv_if_exists(df, filename):
    if os.path.isfile(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        if not os.path.exists(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        df.to_csv(filename, mode="w", header=True, index=False)


def load_storage_and_policy(device, env, hyperparameters, last_model, logdir, n_envs):
    model, observation_shape, policy = initialize_model(device, env, hyperparameters)
    if logdir is not None:
        policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])
    # Test if necessary:
    policy.device = device
    storage = Storage(observation_shape, model.output_dim, hyperparameters["n_steps"], n_envs, device)

    try:
        action_names = get_action_names(env)
    except Exception:
        action_names = None
    obs = env.reset()
    hidden_state = np.zeros((n_envs, storage.hidden_state_size))
    done = np.zeros(n_envs)
    # frames = obs
    policy.eval()
    return action_names, done, hidden_state, obs, policy, storage


def load_hparams_for_model(param_name, logdir, n_envs):
    hyperparameters = get_hyperparams(param_name)
    last_model = latest_model_path(logdir)
    print(last_model)
    hp_file = os.path.join(GLOBAL_DIR, logdir, "hyperparameters.npy")
    if os.path.exists(hp_file):
        hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    if n_envs is not None:
        hyperparameters["n_envs"] = n_envs
    return hyperparameters, last_model


def latest_model_path(logdir, prefix="model"):
    logdir = os.path.join(GLOBAL_DIR, logdir)
    files = os.listdir(logdir)
    pattern = rf"{prefix}_(\d*)\.pth"
    checkpoints = [int(re.search(pattern, x).group(1)) for x in files if re.search(pattern, x)]
    last_model = os.path.join(logdir, f"{prefix}_{max(checkpoints)}.pth")
    return last_model


def floats_to_dp(s, decimals=2):
    replace_code = "REPLACE_CODE"

    flts = re.findall("(\d+\.\d+)", s)
    flts = [f"{float(flt):.{decimals}f}" for flt in flts]
    s = re.sub("(\d+\.\d+)", replace_code, s)
    for flt in flts:
        s = re.sub(replace_code, flt, s, 1)
    return s


def wandb_login():
    wandb.login(key="cfc00eee102a1e9647b244a40066bfc5f1a96610")


class DictToArgs:
    def __init__(self, input_dict):
        for key in input_dict.keys():
            setattr(self, key, input_dict[key])


def add_symbreg_args(parser):
    parser.add_argument('--data_size', type=int, default=100, help='How much data to train on')
    parser.add_argument('--maxsize', type=int, default=20, help='Max equation size')
    parser.add_argument('--iterations', type=int, default=1, help='How many genetic algorithm iterations')
    parser.add_argument('--logdir', type=str, default=None, help='Dir of model to imitate')
    parser.add_argument('--n_envs', type=int, default=int(8),
                        help='Number of parallel environments to use to generate data and test models')
    parser.add_argument('--rounds', type=int, default=int(10), help='Number of episodes to test models for')
    parser.add_argument('--binary_operators', type=str, nargs='+', default=["+", "-", "*", "greater"],
                        help="Binary operators to use in search")
    parser.add_argument('--unary_operators', type=str, nargs='+', default=["square", "exp", "relu"],
                        help="Unary operators to use in search")
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help="Tags for Weights & Biases")
    parser.add_argument('--wandb_name', type=str, default=None, help='Experiment Name for Weights & Biases')
    parser.add_argument('--wandb_group', type=str, default=None)

    parser.add_argument('--timeout_in_seconds', type=int, default=36000)
    parser.add_argument('--populations', type=int, default=24)
    parser.add_argument('--procs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--ncycles_per_iteration', type=int, default=550)
    parser.add_argument('--model_selection', type=str, default="best")
    parser.add_argument('--loss_function', type=str, default="mse")
    parser.add_argument('--weight_metric', type=str, nargs='+', default=None)

    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--bumper', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')

    parser.add_argument('--no-stochastic', dest='stochastic', action='store_false')
    parser.add_argument('--no-bumper', dest='bumper', action='store_false')
    parser.add_argument('--no-denoise', dest='denoise', action='store_false')
    parser.add_argument('--no-use_wandb', dest='use_wandb', action='store_false')

    parser.set_defaults(stochastic=True, bumper=True, denoise=True, use_wandb=True)

    return parser


def inverse_sigmoid(p):
    # this bit makes sure no probabilities are at 1 or 0
    min_val = np.finfo(p.dtype).resolution
    p[p == 0] += min_val
    p[p == 1] -= min_val
    z = np.log(p / (1 - p))
    return z


def sigmoid(h):
    p = 1 / (1 + np.exp(-h))
    return p


def run_subprocess(cmd, newline, suppress=False, timeout=-1):
    # timed = timeout > 0

    if suppress:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True,
                             stderr=subprocess.DEVNULL)  # , start_new_session=timed)
    else:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)  # , start_new_session=timed)
    # if timed:
    #     try:
    #         p.wait(timeout=timeout)
    #     except subprocess.TimeoutExpired:
    #         print("Subprocess Timeout")
    #         os.kill(p.pid, signal.CTRL_C_EVENT)
    #         subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
    #         return "Timeout"
    output = p.communicate()[0]

    output = '\n'.join(output.decode("utf-8").split(newline))
    return output


def sample_from_sigmoid(p):
    return np.int32(np.random.random(p.shape) < p)


def free_gpu(remove_dict):
    remove_list = [f"{x}.doc.ic.ac.uk" for x in remove_dict.keys()]

    LABCOMPS = "^(ash|beech|cedar|curve|gpu|line|oak|pixel|ray|texel|vertex|willow)[0-9]{2}\.doc"

    # Get condor machine status
    # We restrict to lab machines only
    condor_status = subprocess.run(
        [
            "/usr/local/condor/release/bin/condor_status",
            "-long",
            "-attributes",
            "Machine,LoadAvg,State,GPUsMemoryUsage,AssignedGPUs,CUDADeviceName,CUDAGlobalMemoryMb,CUDACoresPerCU",
            "-json",
            "-constraint",
            'regexp( \"' + LABCOMPS + '\", Machine)',
        ],
        stdout=subprocess.PIPE,
        check=True,  # Raises exception if subprocess fails
    )

    # Collect machine information,
    machines = json.loads(condor_status.stdout)
    # x = {m["Machine"]:m["CUDADeviceName"] for m in machines if re.search("RTX 40",m["CUDADeviceName"])}
    # y =  {m["Machine"]:m["CUDADeviceName"] for m in machines if m["CUDAGlobalMemoryMb"]>15000}
    # all_cuda = {m["Machine"]:m["CUDADeviceName"] for m in machines if "CUDADeviceName" in m.keys()}
    # x = {m:all_cuda[m] for m in all_cuda.keys() if re.search("RTX 40",all_cuda[m])}
    # x = {m["Machine"]:[m["State"],m["LoadAvg"],m["GPUsMemoryUsage"]] for m in machines if re.search("RTX 40",m["CUDADeviceName"])}

    # Should we change filter to < 1000 GPUsMemoryUsage?

    # CUDACapability
    # CUDAClockMhz
    # CUDAComputeUnits
    # CUDACoresPerCU
    # CUDADeviceName
    # CUDADevicePciBusId
    # CUDADeviceUuid
    # CUDADriverVersion
    # CUDAECCEnabled
    # CUDAGlobalMemoryMb

    machines = list(filter(lambda m: m["Machine"] not in remove_list, machines))

    machines = list(filter(lambda m: "CUDAGlobalMemoryMb" in m.keys() and "CUDADeviceName" in m.keys(), machines))

    # Rule 0: Pick an RTX 4080 if free
    rtx40 = filter(
        lambda m: re.search("RTX 40", m["CUDADeviceName"]) and m["State"] == "Unclaimed" and m["LoadAvg"] < 0.1,
        machines)

    # else pick an RTX with > 10GB memory (RTX 2080 Ti is typical)
    rtx = filter(lambda m: re.search("RTX", m["CUDADeviceName"]) and m["CUDAGlobalMemoryMb"] > 10000 and m[
        "State"] == "Unclaimed" and m["LoadAvg"] < 0.1, machines)

    # else pick a machine with > 10GB GPU memory
    g10 = filter(lambda m: m["CUDAGlobalMemoryMb"] > 10000 and m["State"] == "Unclaimed" and m["LoadAvg"] < 0.1,
                 machines)

    # Rule 1: Pick a random unclaimed machine with less than 0.1 load average
    # We set less than 0.1 load average to avoid any other process such as root
    # package updates from running in the background that Condor does not pick up
    unclaimed = filter(lambda m: m["State"] == "Unclaimed" and m["LoadAvg"] < 0.1, machines)

    # Rule 2: Pick a random Claimed machine, yield from Condor for interactive usage
    # Interactive usage takes priority over Condor jobs
    claimed = filter(lambda m: m["State"] == "Claimed", machines)

    # Rule 3: Shared usage, all machines are used at the moment, we'll first pick machines
    # that have half the cpu load, i.e. < 4.0 load average
    low_usage = filter(lambda m: m["LoadAvg"] < 0.1, machines)

    # Rule 4: All shared usage, all machines are used and have load average high,
    # we have no option but to randomly pick one for the user
    all_machines = machines

    # Select a machine, yielding from rules in order
    for ms in [rtx40, rtx, g10, unclaimed, claimed, low_usage, all_machines]:
        ms = list(ms)
        if ms:
            labm = random.choice(ms)["Machine"]
            # Check if we can reach the machine, is it running?
            pingt = subprocess.run(
                ["ping", "-c", "1", "-W", "1", labm],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            if pingt.returncode == 0:
                return labm


def entropy_from_binary_prob(p):
    return - (p * np.log(p)) - ((1 - p) * np.log(1 - p))


def get_saved_hyperparams(logdir):
    hp_file = os.path.join(GLOBAL_DIR, logdir, "hyperparameters.npy")
    hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    return hyperparameters


def softmax(Y):
    l = np.exp(Y)
    l[np.isinf(l)] = np.finfo(l.dtype).max
    return l / np.repeat(l.sum(1), Y.shape[-1]).reshape(l.shape)


def sample_numpy_probs(p):
    r = np.random.random(p.shape[0]).repeat(p.shape[-1]).reshape(p.shape)
    from_last = (p.cumsum(1) > r).sum(1)
    nones = from_last == 0
    if np.any(nones):
        from_last[nones] = np.random.random_integers(1, p.shape[-1], sum(nones))
    Y_act = p.shape[-1] - from_last
    return Y_act


def round_to_nearest_in(a, b):
    return b[match_to_nearest(a, b)]


def match_to_nearest(a, b):
    n = len(b)
    shp = list(a.shape) + [n]

    broad = np.repeat(a, n).reshape((shp))
    diffs = np.abs(broad - b)

    return diffs.argmin(-1)


def get_attributes(env):
    vn = env.__dir__()
    vn = [v for v in vn if not re.search("__", v)]


def concat_np_list(l, shape):
    output = np.repeat("", np.prod(shape)).reshape(shape)
    for arr in l:
        if type(arr) == np.ndarray:
            arr = arr.astype(str)
        output = np.core.defchararray.add(output, arr)
    return output


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_project(env_name, exp_name):
    if env_name == "boxworld":
        project = "Box-World"
    elif exp_name == "coinrun-hparams":
        project = "Hparams Coinrun"
    elif exp_name == "coinrun-grok":
        project = "Coinrun Grok"
    elif env_name == "coinrun":
        project = "Coinrun VQMHA"
    elif env_name == "cartpole":
        project = "CartPole"
        if exp_name == "symbreg":
            project = "Graph Symb Reg"
    elif exp_name == "mountain_car_cont_rew":
        project = "MountainCar Continuous Reward"
    elif exp_name == "espo-graph":
        project = "ESPO Graph RL"
    else:
        project = env_name
    if exp_name == "overfit":
        project = "Overfit"
    return project


def get_logdir_from_symbdir(symbdir):
    return re.search(r"(logs.*)symbreg", symbdir).group(1)


def scp(symbdir):
    cmd = ["scp.sh", symbdir]
    run_subprocess(cmd, "\\n")
    print("stop")


def get_agent_constructor(algo):
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    elif algo == 'ppo-model':
        from agents.ppo_model import PPOModel as AGENT
    elif algo == 'graph-agent':
        from agents.graph_agent import GraphAgent as AGENT
    elif algo == 'double-graph-agent':
        from agents.double_graph_agent import DoubleGraphAgent as AGENT
    elif algo == 'ppo-pure':
        from agents.ppo_pure import PPOPure as AGENT
    elif algo == 'espo':
        from agents.espo import ESPO as AGENT
    elif algo == 'IPL':
        from agents.IPL import IPL as AGENT
    elif algo == 'IPL_ICM':
        from agents.IPL_ICM import IPL_ICM as AGENT
    elif algo == 'goal-seeker':
        from agents.goal_seeker import GoalSeeker as AGENT
    elif algo == 'sae':
        from agents.sae import SAE as AGENT
    else:
        raise NotImplementedError
    return AGENT


def add_pets_args(parser):
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--env_name', type=str, default='cartpole_continuous', help='environment ID')
    # parser.add_argument('--val_env_name', type=str, default=None, help='optional validation environment ID')
    # parser.add_argument('--param_name', type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--num_checkpoints', type=int, default=4, help='Number of save checkpoints')
    parser.add_argument('--trial_length', type=int, default=200, help='Length of each trial')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of linear layers in each MLP of dynamics model')
    parser.add_argument('--hid_size', type=int, default=256, help='Hidden linear layer width')
    parser.add_argument('--planning_horizon', type=int, default=15, help='Rollout length')
    parser.add_argument('--replan_freq', type=int, default=1, help='Replan frequency')
    parser.add_argument('--num_iterations', type=int, default=5, help='CEM iterations')
    parser.add_argument('--population_size', type=int, default=500, help='CEM population size')
    parser.add_argument('--num_particles', type=int, default=20, help='Trajectory optimization particles')

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=50)

    parser.add_argument('--validation_ratio', type=float, default=0.05, help='Validation ratio for training')
    parser.add_argument('--elite_ratio', type=float, default=0.1, help='CEM elite ratio')
    parser.add_argument('--alpha', type=float, default=0.1, help='CEM alpha')

    #
    parser.add_argument('--model_batch_size', type=int, default=32)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs='+')
    parser.add_argument('--dyn_model', type=str, default='pets.pets_models.GraphTransitionPets')

    parser.add_argument('--detect_nan', action="store_true")
    parser.add_argument('--deterministic', action="store_true")
    # parser.add_argument('--use_valid_env', action="store_true")
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--overfit', action="store_true")
    parser.add_argument('--use_custom_reward_fn', action="store_true")
    parser.add_argument('--residual', action="store_true")
    #
    parser.add_argument('--no-deterministic', dest='deterministic', action="store_false")
    parser.add_argument('--no-detect_nan', dest='detect_nan', action="store_false")
    parser.add_argument('--no-render', dest='render', action="store_false")
    parser.add_argument('--no-overfit', dest='overfit', action="store_false")
    parser.add_argument('--no-use_custom_reward_fn', dest='use_custom_reward_fn', action="store_false")
    parser.add_argument('--no-residual', dest='residual', action="store_false")

    # parser.add_argument('--no-use_valid_env', dest='use_valid_env', action="store_false")
    parser.add_argument('--no-use_wandb', dest='use_wandb', action="store_false")

    parser.set_defaults(
        deterministic=False,
        detect_nan=False,
        render=False,
        overfit=False,
        use_custom_reward_fn=True,
        # use_valid_env=True,
        use_wandb=True,
        residual=False,
    )

    return parser


def initialize_storage(args, device, double_graph, hidden_state_dim, model_based, n_envs, n_steps, observation_shape,
                       continuous_actions=False, act_shape=None, IPL=False, goal_seeker=False, algo=None):
    create_storage = lambda: Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device, continuous_actions,
                                     act_shape)

    if model_based or double_graph:
        create_storage = lambda: BasicStorage(observation_shape, n_steps, n_envs, device)
    if IPL:
        create_storage = lambda: IPLStorage(observation_shape, n_steps, n_envs, device)
    if goal_seeker:
        create_storage = lambda: GoalSeekerStorage(observation_shape, n_steps, n_envs, device, continuous_actions,
                                                   act_shape)
    if algo == "sae":
        create_storage = lambda: SAEStorage(observation_shape, hidden_state_dim, n_steps, n_envs, device,
                                            continuous_actions, act_shape)
    storage = create_storage()
    storage_valid = create_storage() if args.use_valid_env else None
    storage_greedy = create_storage() if args.use_greedy_env else None
    return storage, storage_valid, storage_greedy


def print_dict(log):
    for k, v in log.items():
        print(f"{k}:\t{v}")


def add_espl_args(parser):
    parser.add_argument('--arch_index', default=0, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--data_size', default=1000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--sample_num', default=2, type=int)
    parser.add_argument('--hard_gum', default=True, type=bool)
    parser.add_argument('--data_scale', default=20, type=int)
    parser.add_argument('--wandb_tags', nargs='+', type=str)
    parser.add_argument('--other_loss_scale', default=100., type=float)
    parser.add_argument('--hard_ratio', default=.7, type=float)
    parser.add_argument('--dist_func', default="mse", type=str)

    parser.add_argument('--target_ratio', default=0.001, type=float)
    parser.add_argument('--spls', default=0.1, type=float)
    parser.add_argument('--constrain_scale', default=1, type=int)
    parser.add_argument('--l0_scale', default=0.01, type=float)
    parser.add_argument('--bl0_scale', default=0, type=int)
    parser.add_argument('--target_temp', default=0.2, type=float)
    parser.add_argument('--warmup_epoch', default=0, type=int)

    return parser


def create_symb_dir_if_exists(logdir, dir_name="symbreg"):
    save_file = "symb_reg.csv"
    symbdir = os.path.join(logdir, dir_name)
    if not os.path.exists(symbdir):
        os.mkdir(symbdir)
    symbdir = os.path.join(symbdir, time.strftime("%Y-%m-%d__%H-%M-%S"))
    if not os.path.exists(symbdir):
        os.mkdir(symbdir)
    return symbdir, save_file
