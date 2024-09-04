import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv


def create_humanoid():
    n_envs = 2
    envs = gym.vector.make('Humanoid-v4', num_envs=n_envs)
    _ = envs.reset(seed=42)
    actions = envs.action_space.sample()
    observations, rewards, termination, truncation, infos = envs.step(actions)


class GymnasiumEnv(gym.Env):
    def __init__(self, env_name, n_envs):
        self.env = gym.vector.make(env_name, num_envs=n_envs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, actions):
        obs, rew, done, trunc, infos = self.env.step(actions)
        # if np.any(trunc):
        #     print("?")
        done |= trunc # TODO: correct?
        return obs, rew, done, infos

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)[0]



if __name__ == "__main__":
    env = GymnasiumEnv('Humanoid-v4', 2)
    obs = env.reset(seed=42)
    for _ in range(1000):
        actions = env.env.action_space.sample()
        obs, rew, done, info = env.step(actions)

    print(obs.shape)
