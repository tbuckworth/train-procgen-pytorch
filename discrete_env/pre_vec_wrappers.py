import gymnasium as gym
import numpy as np


class PetsWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, False, info

    def reset(self, seed=None):
        return self.env.reset(seed=seed), {}

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def done_func(self, act, obs):
        return self.env.done_func(obs)

    def rew_func(self, act, obs):
        return self.env.rew_func(obs)


class DeVecEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.n_envs = self.env.n_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        # is adding a dimension wrong?
        shp = [self.n_envs] + [1 for _ in action.shape]
        action = np.tile(action, shp)
        obs, rew, done, info = self.env.step(action)
        return obs[0], rew[0], done[0], info[0]

    def reset(self, seed=None):
        return self.env.reset(seed=seed)[0]

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def done_func(self, state):
        return self.env.done_func(state)[0]

    def rew_func(self, state):
        return self.env.rew_func(state)[0]