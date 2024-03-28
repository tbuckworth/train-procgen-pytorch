import os
import numpy as np
import pandas as pd
import sympy

from cartpole.create_cartpole import create_cartpole

if os.name != "nt":
    from pysr import PySRRegressor
# Important! keep torch after pysr
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from common.env.procgen_wrappers import create_env
from helper import get_config, get_path, balanced_reward, GLOBAL_DIR, append_to_csv_if_exists, load_storage_and_policy, \
    load_hparams_for_model
from inspect_agent import load_policy


# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\PycharmProjects\train-procgen-pytorch\venv\julia_env\pyjuliapkg\install\bin"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\AppData\Local\Microsoft\WindowsApps"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\.julia\juliaup\julia-1.10.0+0.x64.w64.mingw32\bin"

def find_model(X, Y, symbdir, iterations, save_file):
    model = PySRRegressor(
        equation_file=get_path(symbdir, save_file),
        niterations=iterations,  # < Increase me for better results
        binary_operators=["+", "*", "greater"],
        unary_operators=[
            # "cos",
            # "exp",
            # "sin",
            # "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        denoise=True,
        elementwise_loss="L2MarginLoss()",
        extra_sympy_mappings={"greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True))}
        # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
    print("fitting model")
    model.fit(X, Y)
    print("fitted")
    print(model)
    return model


def load_nn_policy(logdir, n_envs=2):
    cfg = get_config(logdir)

    if cfg["env_name"] == "coinrun":
        action_names, done, env, hidden_state, obs, policy, storage = load_policy(False, logdir,
                                                                                  n_envs=n_envs,
                                                                                  hparams=cfg["param_name"],
                                                                                  start_level=cfg["start_level"],
                                                                                  num_levels=cfg["num_levels"])
        sampler = sample_latent_output_fsqmha
        symbolic_agent_constructor = NeuroSymbolicAgent
        test_env = get_coinrun_test_env(logdir, n_envs)
        test_agent = test_coinrun_agent
    if cfg["env_name"] == "cartpole":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hyperparameters, last_model = load_hparams_for_model(cfg["param_name"], logdir, n_envs)
        env = create_cartpole(n_envs, hyperparameters, is_valid=False)

        action_names, done, hidden_state, obs, policy, storage = load_storage_and_policy(device, env, hyperparameters,
                                                                                         last_model, logdir, n_envs)
        sampler = sample_latent_output_mlpmodel
        symbolic_agent_constructor = SymbolicAgent
        test_env = create_cartpole(n_envs, hyperparameters, is_valid=True)
        test_agent = test_cartpole_agent
    return policy, env, sampler, symbolic_agent_constructor, test_env, test_agent


def drop_first_dim(arr):
    shp = np.array(arr.shape)
    new_shape = tuple(np.concatenate(([np.prod(shp[:2])], shp[2:])))
    return arr.reshape(new_shape)


def generate_data(policy, sampler, env, n):
    observation = env.reset()
    x, y, act = sampler(policy, observation)
    X = x
    Y = y
    while len(X) < n:
        observation, rew, done, info = env.step(act)
        x, y, act = sampler(policy, observation)
        X = np.append(X, x, axis=0)
        Y = np.append(Y, y, axis=0)
    return X, Y


def sample_latent_output_fsqmha(policy, observation):
    with torch.no_grad():
        obs = torch.FloatTensor(observation).to(policy.device)
        x = policy.embedder.forward_to_pool(obs)
        h = policy.embedder.forward_from_pool(x)
        dist, value = policy.hidden_to_output(h)
        y = dist.logits.detach().cpu().numpy()
        act = dist.sample()
    return x.cpu().numpy(), y, act.cpu().numpy()


def sample_latent_output_mlpmodel(policy, observation):
    with torch.no_grad():
        x = torch.FloatTensor(observation).to(policy.device)
        h = policy.embedder(x)
        dist, value = policy.hidden_to_output(h)
        y = dist.logits.detach().cpu().numpy()
        act = dist.sample().cpu().numpy()

    return observation, act, act

def test_cartpole_agent(agent, env, print_name, n=40):
    episodes = 0
    obs = env.reset()
    act = agent.forward(obs)
    episode_rewards = []
    while episodes < n:
        ep_reward = env.env.n_steps.copy()
        obs, rew, done, new_info = env.step(act)
        act = agent.forward(obs)
        if np.any(done):
            episodes += np.sum(done)
            episode_rewards += list(ep_reward[done])
    print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
    return np.mean(episode_rewards)


def test_coinrun_agent(agent, env, print_name, n=40):
    performance_track = {}
    episodes = 0
    obs = env.reset()
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


class SymbolicAgent:
    def __init__(self, model, policy):
        self.model = model
        self.policy = policy

    def forward(self, observation):
        with torch.no_grad():
            # obs = torch.FloatTensor(observation).to(self.policy.device)
            h = self.model.predict(observation)
            return np.round(h,0)
            # TODO: do this with numpy instead of torch
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


def get_coinrun_test_env(logdir, n_envs):
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


def create_symb_dir_if_exists(logdir):
    save_file = "symb_reg.csv"
    symbdir = os.path.join(logdir, "symbreg")
    if not os.path.exists(symbdir):
        os.mkdir(symbdir)
    return symbdir, save_file


if __name__ == "__main__":
    iterations = 100
    data_size = 10000
    rounds = 300
    n_envs = 128
    # logdir = "logs/train/coinrun/coinrun/2024-02-20__18-02-16__seed_6033"
    logdir = "logs/train/cartpole/cartpole/2024-03-28__11-27-12__seed_6033"
    symbdir, save_file = create_symb_dir_if_exists(logdir)
    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, n_envs)
    X, Y = generate_data(policy, sampler, env, n=int(data_size))
    print("data generated")
    if os.name != "nt":
        model = find_model(X, Y, symbdir, iterations, save_file)
        ns_agent = symbolic_agent_constructor(model, policy)
        nn_agent = NeuralAgent(policy)
        ns_score_train = test_agent(ns_agent, env, "NeuroSymb Train", rounds)
        nn_score_train = test_agent(nn_agent, env, "Neural    Train", rounds)

        ns_score_test = test_agent(ns_agent, test_env, "NeuroSymb  Test", rounds)
        nn_score_test = test_agent(nn_agent, test_env, "Neural     Test", rounds)

        values = [iterations, data_size, rounds, nn_score_train, ns_score_train, nn_score_test, ns_score_test, logdir]
        columns = ["iterations", "data_size", "rounds", "Neural_score_Train", "NeuroSymb_score_Train",
                   "Neural_score_Test", "NeuroSymb_score_Test", "logdir"]
        df = pd.DataFrame(columns=columns)
        df.loc[0] = values
        append_to_csv_if_exists(df, os.path.join(symbdir, "results.csv"))
