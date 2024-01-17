import os

import numpy as np
import torch

from boxworld.create_box_world import create_box_world_env
from common.storage import Storage
from helper import initialize_model
from inspect_agent import latest_model_path, predict

if __name__ == "__main__":
    logdir = "logs/train/boxworld/boxworld/2024-01-16__12-29-10__seed_6033"

    last_model = latest_model_path(logdir)
    device = torch.device('cpu')
    hp_file = os.path.join(logdir, "hyperparameters.npy")
    if os.path.exists(hp_file):
        hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    n_envs = 2
    env_args = {"n_envs": n_envs,
                "n": hyperparameters.get('grid_size', 12),
                "goal_length": hyperparameters.get('goal_length', 5),
                "num_distractor": hyperparameters.get('num_distractor', 0),
                "distractor_length": hyperparameters.get('distractor_length', 0),
                "max_steps": 10 ** 3,
                "seed": 6033,
                }
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = create_box_world_env(env_args, render=True, normalize_rew=normalize_rew)
    num_actions = env.action_space.n

    model, observation_shape, policy = initialize_model(device, env, hyperparameters)
    policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])
    policy.to(device)
    policy.device = device
    storage = Storage(observation_shape, model.output_dim, 256, n_envs, device)
    obs = env.reset()
    hidden_state = np.zeros((n_envs, storage.hidden_state_size))
    done = np.zeros(n_envs)
    # frames = obs
    policy.eval()

    while True:
        act, log_prob_act, value, hidden_state, pi = predict(policy, obs, hidden_state, done)
        obs, rew, done, info = env.step(act)
