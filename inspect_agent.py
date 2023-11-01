import os
import re

import numpy as np
import pandas as pd
import torch
from gym3 import ToBaselinesVecEnv, ViewerWrapper
from procgen import ProcgenGym3Env

from common.env.procgen_wrappers import VecExtractDictObs, VecNormalize, TransposeFrame, ScaledFloatFrame
from common.storage import Storage
from helper import get_hyperparams, initialize_model, print_values_actions, get_action_names


def predict(policy, obs, hidden_state, done):
    with torch.no_grad():
        obs = torch.FloatTensor(obs).to(device=policy.device)
        hidden_state = torch.FloatTensor(hidden_state).to(device=policy.device)
        mask = torch.FloatTensor(1 - done).to(device=policy.device)
        dist, value, hidden_state = policy(obs, hidden_state, mask)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

    pi = torch.nn.functional.softmax(dist.logits, dim=1)

    return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy(), pi.cpu().numpy()

def main(render=True):
    logdir = "logs/train/coinrun/coinrun/2023-10-31__10-49-30__seed_6033"
    # df = pd.read_csv(os.path.join(logdir, "log-append.csv"))
    files = os.listdir(logdir)
    pattern = r"model_(\d*)\.pth"
    checkpoints = [int(re.search(pattern, x).group(1)) for x in files if re.search(pattern, x)]
    last_model = os.path.join(logdir, f"model_{max(checkpoints)}.pth")
    device = torch.device('cpu')

    hyperparameters = get_hyperparams("easy-200")
    hp_file = os.path.join(logdir, "hyperparameters.npy")
    if os.path.exists(hp_file):
        hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    n_envs = 1
    env_args = {"num": n_envs,
                "env_name": "coinrun",
                "start_level": 0,
                "num_levels": 500,
                "paint_vel_info": True,
                "distribution_mode": "hard"}
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = create_env_4render(env_args, render, normalize_rew)

    model, observation_shape, policy = initialize_model(device, env, hyperparameters)

    policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])
    # Test if necessary:
    policy.device = device

    storage = Storage(observation_shape, model.output_dim, 256, n_envs, device)
    action_names = get_action_names(env)

    obs = env.reset()
    hidden_state = np.zeros((n_envs, storage.hidden_state_size))
    done = np.zeros(n_envs)

    policy.eval()
    while True:
        act, log_prob_act, value, next_hidden_state, pi = predict(policy, obs, hidden_state, done)
        print_values_actions(action_names, pi, value)
        next_obs, rew, done, info = env.step(act)
        obs = next_obs
        hidden_state = next_hidden_state
        if done[0]:
            print(f"Level seed: {info[0]['level_seed']}")

def create_env_4render(env_args, render, normalize_rew=True):
    if render:
        env_args["render_mode"] = "rgb_array"
    venv = ProcgenGym3Env(**env_args)
    if render:
        venv = ViewerWrapper(venv, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    venv = VecExtractDictObs(venv, "rgb")
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


if __name__ == "__main__":
    main()
