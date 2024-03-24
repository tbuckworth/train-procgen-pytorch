import os
import random
import re
import time
from collections import deque

import gym
import numpy as np
import pandas as pd
import yaml
import platform
from matplotlib import pyplot as plt

from common.model import NatureModel, ImpalaModel, MHAModel, ImpalaVQModel, ImpalaVQMHAModel, ImpalaFSQModel, ribMHA, \
    ImpalaFSQMHAModel, RibFSQMHAModel
from common.policy import CategoricalPolicy
from moviepy.editor import ImageSequenceClip

GLOBAL_DIR = "/vol/bitbucket/tfb115/train-procgen-pytorch"


def is_wsl(v: str = platform.uname().release) -> int:
    """
    detects if Python is running in WSL
    """
    if v.endswith("-Microsoft"):
        return 1
    elif v.endswith("microsoft-standard-WSL2"):
        return 2
    return 0

if os.name == "nt":
    GLOBAL_DIR = "C:/Users/titus/PycharmProjects/train-procgen-pytorch/"
if is_wsl() == 2:
    GLOBAL_DIR = "/mnt/c/Users/titus/PycharmProjects/train-procgen-pytorch/"
try:
    if os.getlogin() == "titus":
        GLOBAL_DIR = "/home/titus/PycharmProjects/train-procgen-pytorch/"
except Exception:
    pass

def match(a, b):
    a = a.tolist()
    b = b.tolist()
    return np.array([b.index(x) for x in a if x in b])


def save_gif(frames, filename="test.gif", fps=20):
    frames = np.array(frames.transpose(0, 2, 3, 1) * 255, dtype=np.uint8)
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(filename, fps=fps)

def get_path(dir, file):
    return os.path.join(GLOBAL_DIR, dir, file)

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


def print_action_entropy(action_names, pi):
    val_names = [f"env{i}" for i in range(len(pi))]
    df = pd.DataFrame({**{"variables": action_names}, **{f"env{i}": x for i, x in enumerate(pi)}})
    df2 = df.pivot_table(values=val_names, index="variables", aggfunc="sum")
    scaled_entropy = -(df2[val_names] * np.log(df2[val_names])).sum(0) / np.log(len(df2))
    df2.loc["Entropy(%)"] = scaled_entropy
    df2[val_names] = np.asarray(np.round(np.squeeze(df2[val_names]) * 100, 0), dtype=np.int32)
    print(df2)


def match(a, b, dtype=np.int32):
    a = a.tolist()
    b = b.tolist()
    return np.array([b.index(x) for x in a if x in b], dtype=dtype)


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


def initialize_model(device, env, hyperparameters):
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space
    has_vq = False
    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    elif architecture == 'vqmha':
        has_vq = False
        mha_layers = hyperparameters.get("mha_layers", 1)
        n_latents = hyperparameters.get("n_latents", 1)
        latent_dim = hyperparameters.get("latent_dim", 1)
        model = MHAModel(n_latents, latent_dim, mha_layers)
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
        model = ImpalaFSQModel(in_channels=in_channels)
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
        model = ImpalaFSQMHAModel(in_channels, mha_layers, device, observation_shape, reduce,
                                  n_impala_blocks=n_impala_blocks, levels=levels, no_quant=True)
    elif architecture == 'impalaitn':
        mha_layers = hyperparameters.get("mha_layers", 2)
        reduce = hyperparameters.get('pool_direction', 'feature_wise')
        levels = hyperparameters.get('levels')
        n_impala_blocks = hyperparameters.get("n_impala_blocks", 3)
        use_intention = hyperparameters.get("use_intention", True)
        model = ImpalaFSQMHAModel(in_channels, mha_layers, device, observation_shape, reduce,
                                  n_impala_blocks=n_impala_blocks, levels=levels, use_intention=use_intention, no_quant=True)
    elif architecture == 'ribmha':
        model = ribMHA(in_channels, device, observation_shape)
    elif architecture == 'ribfsqmha':
        mha_layers = hyperparameters.get("mha_layers", 2)
        reduce = hyperparameters.get('pool_direction', 'feature_wise')
        levels = hyperparameters.get('levels')
        model = RibFSQMHAModel(in_channels, mha_layers, device, observation_shape, reduce, levels=levels)
    else:
        raise NotImplementedError(f"Architecture:{architecture} not found in helper.py")
    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size, has_vq)
    else:
        raise NotImplementedError
    policy.to(device)
    return model, observation_shape, policy


def create_name_from_dict(prefix, suffix, specifications, exclusions=[]):
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


def create_logdir(args, folder, project, subfolder):
    logdir = os.path.join('logs', folder, project, subfolder)
    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{args.seed}'
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
    parser.add_argument('--use_wandb', action="store_true", default=False)
    parser.add_argument('--real_procgen', action="store_true", default=True)
    parser.add_argument('--mirror_env', action="store_true", default=False)
    parser.add_argument('--mut_info_alpha', type=float, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--n_minibatch', type=int, default=None)
    parser.add_argument('--detect_nan', action="store_true", default=False)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--use_valid_env', action="store_true", default=True)
    parser.add_argument('--render', action="store_true", default=False)
    parser.add_argument('--paint_vel_info', action="store_true", default=True)
    parser.add_argument('--reduce_duplicate_actions', action="store_true", default=True)
    parser.add_argument('--wandb_tags', type=str, nargs='+')
    parser.add_argument('--minibatches', type=int, nargs='+')
    parser.add_argument('--levels', type=int, nargs='+', default=None)

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
        df.to_csv(filename, mode="w", header=True, index=False)
