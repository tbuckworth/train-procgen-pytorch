import gymnasium as gym
import numpy as np
import torch


class PetsWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def get_customizable_params(self):
        return self.env.get_customizable_params()

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
        done = self.env.done_func(obs.detach().cpu().numpy())
        return torch.BoolTensor(done).to(device=obs.device).unsqueeze(-1)
    def rew_func(self, act, obs):
        rew = self.env.rew_func(obs.detach().cpu().numpy())
        return torch.Tensor(rew).type(obs.dtype).to(device=obs.device)


class DeVecEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.n_envs = self.env.n_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def get_customizable_params(self):
        return self.env.customizable_params

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
        return self.env.done_func(state)

    def rew_func(self, state):
        return self.env.rew_func(state)