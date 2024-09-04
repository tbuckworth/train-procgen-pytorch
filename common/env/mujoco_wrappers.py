import gymnasium as gym
import numpy as np


def create_humanoid():
    n_envs = 2
    envs = gym.vector.make('Humanoid-v4', num_envs=n_envs)
    _ = envs.reset(seed=42)
    actions = np.array([1, 0, 1])
    observations, rewards, termination, truncation, infos = envs.step(actions)
