import gym3
import gym
import numpy as np
from gym3 import ToBaselinesVecEnv, ViewerWrapper

from common.env.procgen_wrappers import VecNormalize


class GymWrapper(gym.Wrapper):
    def seed(self, seed):
        return self.reset(seed=seed)

def create_baselines_vec_from_gym(env_name="CartPole-v1", num=2):
    # Requires gym==0.25.2 (0.26.2 won't work!)
    def create_cartpole():
        return GymWrapper(gym.make(env_name))
    envs = gym3.vectorize_gym(num=num, env_fn=create_cartpole)
    venv = ToBaselinesVecEnv(envs)
    return venv


def gym_test():
    env_args = {"env_name": "CartPole-v1",
                "n_envs": 2}
    venv = create_env_gym(env_args, render=False)


def create_env_gym(env_args, render, normalize_rew=True):
    n_envs = env_args["n_envs"]
    venv = create_baselines_vec_from_gym(env_name=env_args["env_name"], num=n_envs)
    if render:
        venv = ViewerWrapper(venv, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    # TODO: check if we need this:
    # venv = VecExtractDictObs(venv, "rgb")
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    # venv = TransposeFrame(venv)
    # venv = ScaledFloatFrame(venv)
    return venv

if __name__ == "__main__":
    gym_test()
