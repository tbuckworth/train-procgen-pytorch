import os

import numpy as np
import torch

from boxworld.create_box_world import create_box_world_env
from common.storage import Storage
from helper import initialize_model, last_folder, print_values_actions
from inspect_agent import latest_model_path, predict


def run_boxworld(use_valid_env=False):
    # global logdir, device, n_envs, env, model, policy, policy
    env_suffix = ""
    if use_valid_env:
        env_suffix = "_v"
    logdir = last_folder("logs/train/boxworld/boxworld")
    last_model = latest_model_path(logdir)
    device = torch.device('cpu')
    hp_file = os.path.join(logdir, "hyperparameters.npy")
    if os.path.exists(hp_file):
        hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    n_envs = 2
    env_args = {"n_envs": n_envs,
                "n": hyperparameters.get(f'grid_size{env_suffix}', 12),
                "goal_length": hyperparameters.get(f'goal_length{env_suffix}', 5),
                "num_distractor": hyperparameters.get(f'num_distractor{env_suffix}', 0),
                "distractor_length": hyperparameters.get(f'distractor_length{env_suffix}', 0),
                "max_steps": 10 ** 3,
                "seed": None,
                }
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = create_box_world_env(env_args, render=True, normalize_rew=normalize_rew)
    num_actions = env.action_space.n
    # TODO: check this is correct:
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    # env.venv.get_action_lookup()
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
    time_steps = 0
    while True:
        act, log_prob_act, value, hidden_state, pi = predict(policy, obs, hidden_state, done)
        print_values_actions(action_names, pi, value, i=action_names[act[0]])
        obs, rew, done, info = env.step(act)
        time_steps += 1
        if done[0]:
            print(f"Done. Reward:{rew[0]:.2f}. Timesteps: {time_steps}")
            time_steps = 0


if __name__ == "__main__":
    run_boxworld(True)
