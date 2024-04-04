import argparse
import os
import parser

import numpy as np
import pandas as pd
import sympy

from cartpole.create_cartpole import create_cartpole
from email_results import send_image

if os.name != "nt":
    from pysr import PySRRegressor
# Important! keep torch after pysr
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from common.env.procgen_wrappers import create_env
from helper import get_config, get_path, balanced_reward, GLOBAL_DIR, load_storage_and_policy, \
    load_hparams_for_model, floats_to_dp, append_to_csv_if_exists
from inspect_agent import load_policy
from matplotlib import pyplot as plt


# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\PycharmProjects\train-procgen-pytorch\venv\julia_env\pyjuliapkg\install\bin"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\AppData\Local\Microsoft\WindowsApps"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\.julia\juliaup\julia-1.10.0+0.x64.w64.mingw32\bin"

def find_model(X, Y, symbdir, iterations, save_file, weights, args):
    model = PySRRegressor(
        equation_file=get_path(symbdir, save_file),
        niterations=iterations,  # < Increase me for better results
        binary_operators=args.binary_operators,
        unary_operators=args.unary_operators,
        weights=weights,
        denoise=True,
        extra_sympy_mappings={"greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True))},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
    print("fitting model")
    model.fit(X, Y)
    eqn_str = get_best_str(model)
    print(eqn_str)
    return model


def get_best_str(model, split='\n'):
    best = model.get_best()
    if type(best) == list:
        return split.join([x.equation for x in model.get_best()])
    return model.get_best().equation


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
    x, y, act, value = sampler(policy, observation)
    X = x
    Y = y
    V = value
    while len(X) < n:
        observation, rew, done, info = env.step(act)
        x, y, act, v = sampler(policy, observation)
        X = np.append(X, x, axis=0)
        Y = np.append(Y, y, axis=0)
        V = np.append(V, v, axis=0)
    return X, Y, V


def sample_latent_output_fsqmha(policy, observation):
    with torch.no_grad():
        obs = torch.FloatTensor(observation).to(policy.device)
        x = policy.embedder.forward_to_pool(obs)
        h = policy.embedder.forward_from_pool(x)
        dist, value = policy.hidden_to_output(h)
        y = dist.logits.detach().cpu().numpy()
        act = dist.sample()
    return x.cpu().numpy(), y, act.cpu().numpy(), value.cpu().numpy()


def sample_latent_output_mlpmodel(policy, observation):
    with torch.no_grad():
        x = torch.FloatTensor(observation).to(policy.device)
        h = policy.embedder(x)
        dist, value = policy.hidden_to_output(h)
        y = dist.logits.detach().cpu().numpy()
        # act = dist.sample().cpu().numpy()
        act = y.argmax(axis=1)
    return observation, act, act, value.cpu().numpy()


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


class AnalyticModel:
    def forward(self, observation):
        out = np.ones((observation.shape[0],))
        term = 3 * observation[:, 2] + observation[:, 3]
        out[term <= 0] = 0
        return out

    def predict(self, observation):
        return self.forward(observation)


class SymbolicAgent:
    def __init__(self, model, policy):
        self.model = model
        self.policy = policy

    def forward(self, observation):
        with torch.no_grad():
            # obs = torch.FloatTensor(observation).to(self.policy.device)
            h = self.model.predict(observation)
            return np.round(h, 0)
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



def send_full_report(df, logdir, model):
    # load log csv
    dfl = pd.read_csv(os.path.join(logdir, "log-append.csv"))
    # create graph
    roll_window = 100
    dfl2 = pd.DataFrame(
        {"Neural Training Reward - Rolling Avg.": dfl["mean_episode_rewards"].rolling(window=roll_window).mean(),
         "Neural Test Reward - Rolling Avg.": dfl["val_mean_episode_rewards"].rolling(window=roll_window).mean()
         })  # ,
    dfl2.index = dfl["timesteps"]

    dfl2.plot()
    ns_train = df.NeuroSymb_score_Train[0]
    plt.hlines(y=ns_train,
               xmin=0,
               xmax=dfl2.index.max(),
               label="NeuroSymbolic Training Reward",
               linestyles="dashed")
    ns_test = df.NeuroSymb_score_Test[0]
    plt.hlines(y=ns_test,
               xmin=0,
               xmax=dfl2.index.max(),
               label="NeuroSymbolic Test Reward",
               linestyles="dashed")
    plt.ylim(ymin=0)  # this line
    plt.legend()
    plot_file = os.path.join(logdir, "training_plot.png")
    plt.savefig(plot_file)
    # create table
    dfv = df.T
    dfv.columns = ["Value"]
    # TODO: make dfv formatted to two dps
    tab_code = dfv.to_html()  # df.T.to_html(columns=False)
    eqn_str = get_best_str(model, split="<br/>")
    # send email
    eqn_str = floats_to_dp(eqn_str)

    nn_test = df.Neural_score_Test[0]
    nn_train = df.Neural_score_Train[0]
    test_improved = ns_test > nn_test
    train_improved = ns_train > nn_train

    statement = "Failed"
    if test_improved and train_improved:
        statement = "Complete Success"
    if not test_improved and train_improved:
        statement = "Improved Training Reward"
    if test_improved and not train_improved:
        statement = "Improved Generalization"

    body_text = f"<b>{statement}</b><br>{tab_code}<br><b>Learned Formula:</b><br><p>{eqn_str}</p>"
    send_image(plot_file, "PySR Results", body_text=body_text)


def temp_func():
    iterations = 10
    data_size = 1000
    rounds = 300
    n_envs = 32
    # Coinrun:
    # logdir = "logs/train/coinrun/coinrun/2024-02-20__18-02-16__seed_6033"

    # Very Sparse Coinrun:
    # logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"

    # # cartpole that didn't work:
    # logdir = "logs/train/cartpole/cartpole/2024-03-28__11-27-12__seed_6033"
    #
    # # 10bn timestep cartpole (works!):
    logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    run_neurosymbolic_search(data_size, iterations, logdir, n_envs, rounds)


def run_neurosymbolic_search(args):  # data_size, iterations, logdir, n_envs, rounds):
    data_size = args.data_size
    iterations = args.iterations
    logdir = args.logdir
    n_envs = args.n_envs
    rounds = args.rounds
    symbdir, save_file = create_symb_dir_if_exists(logdir)
    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, n_envs)
    X, Y, V = generate_data(policy, sampler, env, n=int(data_size))
    print("data generated")
    if os.name != "nt":
        model = find_model(X, Y, symbdir, iterations, save_file, V, args)
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
        send_full_report(df, logdir, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=1000, help='How much data to train on')
    parser.add_argument('--iterations', type=int, default=10, help='How many genetic algorithm iterations')
    parser.add_argument('--logdir', type=str, default=None, help='Dir of model to imitate')
    parser.add_argument('--n_envs', type=int, default=int(0),
                        help='Number of parallel environments to use to generate data and test models')
    parser.add_argument('--rounds', type=int, default=int(500), help='Number of episodes to test models for')
    parser.add_argument('--binary_operators', type=str, nargs='+', default=["+", "-", "greater"],
                        help="Binary operators to use in search")
    parser.add_argument('--unary_operators', type=str, nargs='+', default=[], help="Unary operators to use in search")
    parser.add_argument('--denoise', action="store_true", default=False)

    args = parser.parse_args()

    run_neurosymbolic_search(args)
