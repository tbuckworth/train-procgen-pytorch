import gym
import numpy as np
import pandas as pd
import yaml
from gym3 import ViewerWrapper, ToBaselinesVecEnv
from procgen import ProcgenGym3Env

from common.env.procgen_wrappers import VecExtractDictObs, VecNormalize, TransposeFrame, ScaledFloatFrame
from common.model import NatureModel, ImpalaModel, VQMHAModel, ImpalaVQModel, ImpalaVQMHAModel, ImpalaFSQModel
from common.policy import CategoricalPolicy
from moviepy.editor import ImageSequenceClip


def save_gif(frames, filename="test.gif", fps=20):
    frames = np.array(frames.transpose(0, 2, 3, 1) * 255, dtype=np.uint8)
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(filename, fps=fps)


def print_values_actions(action_names, pi, value, i=""):
    ap = np.squeeze(pi[0])
    df = pd.DataFrame({"variables": action_names, "values": ap})
    df2 = df.pivot_table(values="values", index="variables", aggfunc="sum")
    a_names = df2.index.to_numpy()
    max_len = max([len(x) for x in a_names])
    a_names = np.array([x + ''.join([' ' for _ in range(max_len - len(x))]) for x in a_names])
    action_probs = np.asarray(np.round(np.squeeze(df2.values) * 100, 0), dtype=np.int32)
    idx = np.argsort(action_probs)[::-1][:5]
    top_actions = '\t'.join([f"{x[0]}: {x[1]}" for x in zip(action_probs[idx], a_names[idx])])
    if i != "":
        print(f"{i}:\t{np.squeeze(value[0]):.2f}\t{top_actions}")
    else:
        print(f"{np.squeeze(value[0]):.2f}\t{top_actions}")


def match(a, b):
    a = a.tolist()
    b = b.tolist()
    return np.array([b.index(x) for x in a if x in b])


def get_combos(env):
    if hasattr(env, "combos"):
        return env.combos
    if hasattr(env, "env"):
        return get_combos(env.env)
    raise NotImplementedError("No combos found in env")


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
    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    elif architecture == 'vqmha':
        model = VQMHAModel(in_channels, hyperparameters)
    elif architecture == 'impalavq':
        model = ImpalaVQModel(in_channels=in_channels)
    elif architecture == 'impalavqmha':
        mha_layers = hyperparameters.get("mha_layers", 1)
        use_vq = hyperparameters.get("use_vq", True)
        model = ImpalaVQMHAModel(in_channels=in_channels, mha_layers=mha_layers, device=device, use_vq=use_vq)
    elif architecture == 'impalafsq':
        model = ImpalaFSQModel(in_channels=in_channels)
    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)
    return model, observation_shape, policy


def create_env(env_args, render, normalize_rew=True):
    if render:
        env_args["render_mode"] = "rgb_array"
    venv = ProcgenGym3Env(**env_args)
    if render:
        venv = ViewerWrapper(venv, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    venv = VecExtractDictObs(venv, "rgb")
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


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
