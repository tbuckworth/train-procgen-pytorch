import os

import numpy as np
# import pandas as pd
# from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from common.env.procgen_wrappers import create_env

if os.name != "nt":
    from pysr import PySRRegressor
import torch

from helper import get_config, get_path, balanced_reward, GLOBAL_DIR
from inspect_agent import load_policy

# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\PycharmProjects\train-procgen-pytorch\venv\julia_env\pyjuliapkg\install\bin"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\AppData\Local\Microsoft\WindowsApps"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\.julia\juliaup\julia-1.10.0+0.x64.w64.mingw32\bin"

def find_model(X, Y, logdir, iterations):
    model = PySRRegressor(
        equation_file=get_path(logdir, "symb_reg.csv"),
        niterations=iterations,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )
    print("fitting model")
    model.fit(X, Y)
    print("fitted")
    print(model)
    return model



def load_nn_policy(logdir, n_envs = 2):
    cfg = get_config(logdir)

    action_names, done, env, hidden_state, obs, policy, storage = load_policy(False, logdir, n_envs=n_envs,
                                                                              hparams="hard-500-impalafsqmha",
                                                                              start_level=cfg["start_level"],
                                                                              num_levels=cfg["num_levels"])
    return policy, env, obs, storage


def drop_first_dim(arr):
    shp = np.array(arr.shape)
    new_shape = tuple(np.concatenate(([np.prod(shp[:2])], shp[2:])))
    return arr.reshape(new_shape)

def generate_data(policy, env, observation, n):
    x, y, act = sample_latent_output(policy, observation)
    X = x
    Y = y
    while len(X) < n:
        observation, rew, done, info = env.step(act)
        x, y, act = sample_latent_output(policy, observation)
        X = np.append(X, x, axis=0)
        Y = np.append(Y, y, axis=0)
    return X, Y


def sample_latent_output(policy, observation):
    with torch.no_grad():
        obs = torch.FloatTensor(observation).to(policy.device)
        x = policy.embedder.forward_to_pool(obs)
        h = policy.embedder.forward_from_pool(x)
        dist, value = policy.hidden_to_output(h)
        y = dist.logits.detach().cpu().numpy()
        act = dist.sample()
    return x.cpu().numpy(), y, act.cpu().numpy()


def test_agent(agent, env, obs, print_name, n=40):
    performance_track = {}
    episodes = 0
    act = agent.forward(obs)
    while episodes < n:
        obs, rew, done, info = env.step(act)
        act = agent.forward(obs)
        true_average_reward = balanced_reward(done, info, performance_track)
        if np.any(done):
            episodes += np.sum(done)
            print(f"{print_name}:\tEpisode:{episodes}\tBalanced Reward:{true_average_reward:.2f}")
    return true_average_reward

def sample_policy_with_symb_model(model, policy, observation):
    with torch.no_grad():
        obs = torch.FloatTensor(observation).to(policy.device)
        x = policy.embedder.forward_to_pool(obs)
        h = model(x)
        dist, value = policy.hidden_to_output(h)
        # y = dist.logits.detach().cpu().numpy()
        act = dist.sample()
    return act.cpu().numpy()

class NeuroSymbolicAgent:
    def __init__(self, model, policy):
        self.model = model
        self.policy = policy

    def forward(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            x = self.policy.embedder.forward_to_pool(obs)
            h = self.model.predict(x)
            logits = torch.FloatTensor(h).to(self.policy.device)
            log_probs = F.log_softmax(logits, dim=1)
            p = Categorical(logits=log_probs)
            act = p.sample()
        return act.cpu().numpy()


class NeuralAgent:
    def __init__(self, policy):
        self.policy = policy

    def forward(self, observation):
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.policy.device)
            h = self.policy.embedder(obs)
            dist, value = self.policy.hidden_to_output(h)
            act = dist.sample()
        return act.cpu().numpy()


def get_test_env(logdir, n_envs):
    cfg = get_config(logdir)
    hp_file = os.path.join(GLOBAL_DIR, logdir, "hyperparameters.npy")
    hyperparameters = np.load(hp_file, allow_pickle='TRUE').item()
    hyperparameters["n_envs"] = n_envs
    env_args = {"num": hyperparameters["n_envs"],
                "env_name": "coinrun",
                "start_level": cfg["start_level"] + cfg["num_levels"] + 1,  # 325
                "num_levels": 0,
                "paint_vel_info": True,
                "distribution_mode": "hard"}
    normalize_rew = hyperparameters.get('normalize_rew', True)
    try:
        cfg = get_config(logdir)
        mirror_some = cfg["mirror_env"]
    except Exception:
        mirror_some = True
    env = create_env(env_args, False, normalize_rew, mirror_some)
    return env

if __name__ == "__main__":
    iterations = 40
    data_size = 10000
    rounds = 16
    n_envs = 8
    logdir = "logs/train/coinrun/coinrun/2024-02-20__18-02-16__seed_6033"
    policy, env, obs, storage = load_nn_policy(logdir, n_envs)
    X, Y = generate_data(policy, env, obs, n=data_size)
    print("data generated")
    if os.name != "nt":
        model = find_model(X, Y, logdir, iterations)
        # for i in range(Y.shape[-1]):
        #     print(i)
        #     sm = model.pytorch([i])
        #     print(sm)
        # sm = model.pytorch([i for i in range(Y.shape[-1])])
        # print(len(sm))
        # print(sm)
        ns_agent = NeuroSymbolicAgent(model, policy)
        nn_agent = NeuralAgent(policy)
        ns_score_train = test_agent(ns_agent, env, obs, "NeuroSymb Train", rounds)
        nn_score_train = test_agent(nn_agent, env, obs, "Neural    Train", rounds)

        test_env = get_test_env(logdir, n_envs)

        ns_score_test = test_agent(ns_agent, test_env, obs, "NeuroSymb  Test", rounds)
        nn_score_test = test_agent(nn_agent, test_env, obs, "Neural     Test", rounds)
