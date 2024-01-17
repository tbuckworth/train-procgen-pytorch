import os
import time

import gym
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from common.model import NatureModel, ImpalaModel, VQMHAModel, ImpalaVQModel, ImpalaVQMHAModel, ImpalaFSQModel, ribMHA
from common.policy import CategoricalPolicy
from moviepy.editor import ImageSequenceClip


def match(a, b):
    a = a.tolist()
    b = b.tolist()
    return np.array([b.index(x) for x in a if x in b])


def save_gif(frames, filename="test.gif", fps=20):
    frames = np.array(frames.transpose(0, 2, 3, 1) * 255, dtype=np.uint8)
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(filename, fps=fps)


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

def get_action_names(env):
    action_names = np.array(
        [x[0] + "_" + x[1] if len(x) == 2 else (x[0] if len(x) == 1 else "") for x in get_combos(env)])
    x = np.array(['D', 'A', 'W', 'S', 'Q', 'E'])
    y = np.array(['RIGHT', 'LEFT', 'UP', 'DOWN', 'LEFT_UP', 'RIGHT_UP'])
    action_names[match(x, action_names)] = y[match(action_names, x)]
    return action_names


def get_hyperparams(param_name):
    with open('hyperparams/procgen/config.yml', 'r') as f:
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
        has_vq = True
        model = VQMHAModel(in_channels, hyperparameters)
    elif architecture == 'impalavq':
        has_vq = True
        model = ImpalaVQModel(in_channels=in_channels)
    elif architecture == 'impalavqmha':
        has_vq = True
        mha_layers = hyperparameters.get("mha_layers", 1)
        use_vq = hyperparameters.get("use_vq", True)
        model = ImpalaVQMHAModel(in_channels=in_channels, mha_layers=mha_layers, device=device, use_vq=use_vq, obs_shape=observation_shape)
    elif architecture == 'impalafsq':
        model = ImpalaFSQModel(in_channels=in_channels)
    elif architecture == 'impalafsqmha':
        model = ImpalaFSQModel(in_channels, device, use_mha=True)
    elif architecture == 'ribmha':
        model = ribMHA(in_channels, device)
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
