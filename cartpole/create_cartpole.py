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
    venv = CartPoleVecEnv(n_envs, degrees=env_args["degrees"], h_range=env_args["h_range"])
    venv.ob_space = _space2vt(venv.observation_space)
    venv.ac_space = _space2vt(venv.action_space)
    return venv


