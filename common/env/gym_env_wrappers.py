import gym3
import gym
import numpy as np
from gym3 import ToBaselinesVecEnv

from common.env.procgen_wrappers import VecNormalize


class GymWrapper(gym.Wrapper):
    def seed(self, seed):
        return self.reset(seed=seed)

def create_gym_env(env_name="CartPole-v1"):
    # Requires gym==0.25.2 (0.26.2 won't work!)
    def create_cartpole():
        return GymWrapper(gym.make(env_name))
    envs = gym3.vectorize_gym(num=3, env_fn=create_cartpole)
    venv = ToBaselinesVecEnv(envs)
    return venv
    # venv = VecNormalize(venv, ob=False)


def gym_test():
    pass


if __name__ == "__main__":
    gym_test()
