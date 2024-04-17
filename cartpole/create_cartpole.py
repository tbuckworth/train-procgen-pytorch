from gym3 import ViewerWrapper, ToBaselinesVecEnv
from gym3.interop import _space2vt

from cartpole.cartpole_pre_vec import CartPoleVecEnv
from common.env.procgen_wrappers import VecNormalize
def create_cartpole_env_pre_vec(env_args_in, render, normalize_rew=True):
    env_args = env_args_in.copy()
    n_envs = env_args["n_envs"]
    venv = make_cartpole_vec(env_args, n_envs)
    if render:
        venv = ViewerWrapper(venv, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    return venv


def make_cartpole_vec(env_args, n_envs):
    gravity = env_args.get("gravity", 9.8)
    venv = CartPoleVecEnv(n_envs, degrees=env_args["degrees"], h_range=env_args["h_range"], gravity=gravity)
    venv.ob_space = _space2vt(venv.observation_space)
    venv.ac_space = _space2vt(venv.action_space)
    return venv


def create_cartpole(args, hyperparameters, is_valid=False):
    n_envs = hyperparameters.get('n_envs', 32)
    env_args = {"n_envs": n_envs,
                "env_name": "CartPole-v1",
                "degrees": 12,
                "h_range": 2.4,
                }
    if is_valid:
        env_args["degrees"] = hyperparameters.get("degrees_v", 9)
        env_args["h_range"] = hyperparameters.get("h_range_v", 1.2)
    normalize_rew = hyperparameters.get('normalize_rew', False)
    return create_cartpole_env_pre_vec(env_args, render=False, normalize_rew=normalize_rew)
