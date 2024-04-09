import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import sympy
import wandb

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
    load_hparams_for_model, floats_to_dp, dict_to_html_table, wandb_login, add_symbreg_args
from inspect_agent import load_policy
from matplotlib import pyplot as plt


# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\PycharmProjects\train-procgen-pytorch\venv\julia_env\pyjuliapkg\install\bin"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\AppData\Local\Microsoft\WindowsApps"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\.julia\juliaup\julia-1.10.0+0.x64.w64.mingw32\bin"

def find_model(X, Y, symbdir, iterations, save_file, weights, args):
    model = PySRRegressor(
        equation_file=get_path(symbdir, save_file),
        niterations=args.iterations,  # < Increase me for better results
        binary_operators=args.binary_operators,
        unary_operators=args.unary_operators,
        weights=weights,
        denoise=args.denoise,
        extra_sympy_mappings={"greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True))},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        timeout_in_seconds=args.timeout_in_seconds,
        populations=args.populations,
        procs=args.procs,
        batching=args.data_size > 1000,
        weight_optimize=0.001 if args.ncycles_per_iteration > 550 else 0.0,
        ncycles_per_iteration=args.ncycles_per_iteration,
        bumper=args.bumper,
    )
    print("fitting model")
    start = time.time()
    model.fit(X, Y)
    elapsed = time.time() - start
    eqn_str = get_best_str(model)
    print(eqn_str)
    return model, elapsed


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
        test_agent = test_agent_balanced_reward
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
        # This is for cartpole only!
        ep_reward = env.env.n_steps.copy()
        obs, rew, done, new_info = env.step(act)
        act = agent.forward(obs)
        if np.any(done):
            episodes += np.sum(done)
            episode_rewards += list(ep_reward[done])
    print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
    return np.mean(episode_rewards)


def test_agent_balanced_reward(agent, env, print_name, n=40):
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


def test_agent_mean_reward(agent, env, print_name, n=40):
    episodes = 0
    obs = env.reset()
    act = agent.forward(obs)
    episode_rewards = []
    while episodes < n:
        obs, rew, done, info = env.step(act)
        act = agent.forward(obs)
        if np.any(done):
            episodes += np.sum(done)
            episode_rewards += list(rew[done])
    print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
    return np.mean(episode_rewards)


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


class RandomAgent:
    def __init__(self, n_actions):
        self.actions = np.arange(n_actions)

    def forward(self, observation):
        return np.random.choice(self.actions, size=len(observation))


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
    symbdir = os.path.join(symbdir, time.strftime("%Y-%m-%d__%H-%M-%S"))
    if not os.path.exists(symbdir):
        os.mkdir(symbdir)
    return symbdir, save_file


def send_full_report(df, logdir, model, args):
    # load log csv
    dfl = pd.read_csv(os.path.join(logdir, "log-append.csv"))
    cfg = get_config(logdir)
    fine_tune_moment = None
    if cfg["model_file"] is not None:
        model_dir = re.search(f".*(logs.*)model_\d*.pth", cfg["model_file"]).group(1)
        if os.path.exists(model_dir):
            df0 = pd.read_csv(os.path.join(model_dir, "log-append.csv"))
            fine_tune_moment = df0.timesteps.max()
            dfl.timesteps += fine_tune_moment
            # join df0 and dfl:
            dfl = pd.concat([df0, dfl], ignore_index=True)

    # create graph
    roll_window = 100
    dfl2 = pd.DataFrame(
        {"Neural Train": dfl["mean_episode_rewards"].rolling(window=roll_window).mean(),
         "Neural Test": dfl["val_mean_episode_rewards"].rolling(window=roll_window).mean()
         })  # ,
    dfl2.index = dfl["timesteps"]

    ax = dfl2.plot()
    ns_train = df.NeuroSymb_score_Train[0]
    ns_test = df.NeuroSymb_score_Test[0]
    rn_train = df.Random_score_Train[0]
    rn_test = df.Random_score_Test[0]

    hline_dict = {"NeuroSymb Train": [ns_train, "black"],
                  "NeuroSymb Test": [ns_test, "green"],
                  "Random Train": [rn_train, "darkred"],
                  "Random Test": [rn_test, "red"], }
    for key in hline_dict.keys():
        plt_hline(dfl2, key, hline_dict[key][0], hline_dict[key][1])

    min_y = min(0, dfl["mean_episode_rewards"].min(), dfl["val_mean_episode_rewards"].min())
    max_y = max(0, dfl["mean_episode_rewards"].max(), dfl["val_mean_episode_rewards"].max())
    plt.ylim(ymin=min_y)
    plt.title("Rolling Average Reward")

    if fine_tune_moment is not None:
        plt.vlines(x=fine_tune_moment, ymin=min_y, ymax=max_y, linestyles='--')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                     box.width, box.height * 0.85])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3)

    plot_file = os.path.join(logdir, "training_plot.png")
    plt.savefig(plot_file)

    params = dict_to_html_table(args.__dict__)

    eqn_str = get_best_str(model, split="<br/>")
    # send email
    eqn_str = floats_to_dp(eqn_str)

    nn_test = df.Neural_score_Test[0]
    nn_train = df.Neural_score_Train[0]

    tab_code = pd.DataFrame({"Train": [nn_train, ns_train, rn_train],
                             "Test": [nn_test, ns_test, rn_test]},
                            index=["Neural", "NeuroSymbolic", "Random"], ).round(2).to_html()

    test_improved = ns_test > nn_test
    train_improved = ns_train > nn_train

    statement = "Failed"
    if test_improved and train_improved:
        statement = "Complete Success"
    if not test_improved and train_improved:
        statement = "Improved Training Reward"
    if test_improved and not train_improved:
        statement = "Improved Generalization"

    body_text = f"<br><b>{statement}</b><br>{tab_code}<br><b>Learned Formula:</b><br><p>{eqn_str}</p><br>{params}"
    send_image(plot_file, "PySR Results", body_text=body_text)
    print("")


def split_df_by_index_and_pivot(df):
    dfv = df.filter(regex="_score_").T
    dfv2 = pd.Series(dfv.index).str.split("_score_", expand=True)
    dfv2.columns = ["Model", "Environment"]
    dfv2["mean_reward"] = dfv[0].values.round(decimals=2)
    dfw = dfv2.pivot(columns="Environment", index="Model", values="mean_reward")
    tab_code = dfw.to_html(index=True)


def plt_hline(dfl2, label, ns_train, colour):
    plt.hlines(y=ns_train,
               xmin=0,
               xmax=dfl2.index.max(),
               label=label,
               linestyles="dashed",
               color=colour)


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
    cfg = vars(args)
    np.save(os.path.join(symbdir, "config.npy"), cfg)

    wandb_name = args.wandb_name
    if args.wandb_name is None:
        wandb_name = np.random.randint(1e5)

    if args.use_wandb:
        wandb_login()
        name = wandb_name
        wb_resume = "allow"  # if args.model_file is None else "must"
        project = "Symb Reg"
        if args.wandb_group is not None:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name, group=args.wandb_group)
        else:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name)

    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, n_envs)
    X, Y, V = generate_data(policy, sampler, env, n=int(data_size))
    print("data generated")
    if os.name != "nt":
        pysr_model, elapsed = find_model(X, Y, symbdir, iterations, save_file, V, args)
        ns_agent = symbolic_agent_constructor(pysr_model, policy)
        nn_agent = NeuralAgent(policy)
        rn_agent = RandomAgent(env.action_space.n)

        ns_score_train = test_agent(ns_agent, env, "NeuroSymb Train", rounds)
        nn_score_train = test_agent(nn_agent, env, "Neural    Train", rounds)
        rn_score_train = test_agent(rn_agent, env, "Random    Train", rounds)

        ns_score_test = test_agent(ns_agent, test_env, "NeuroSymb  Test", rounds)
        nn_score_test = test_agent(nn_agent, test_env, "Neural     Test", rounds)
        rn_score_test = test_agent(rn_agent, test_env, "Random     Test", rounds)

        # values = [iterations, data_size, rounds, nn_score_train, ns_score_train, nn_score_test, ns_score_test, logdir]
        # columns = ["iterations", "data_size", "rounds", "Neural_score_Train", "NeuroSymb_score_Train",
        #            "Neural_score_Test", "NeuroSymb_score_Test", "logdir"]
        best_loss = pysr_model.get_best().loss
        best_complexity = pysr_model.get_best().complexity
        problem_name = re.search("logs/train/([^/]*)/", logdir).group(1)

        df_values = {
            "Random_score_Train": [rn_score_train],
            "Neural_score_Train": [nn_score_train],
            "NeuroSymb_score_Train": [ns_score_train],
            "Neural_score_Test": [nn_score_test],
            "Random_score_Test": [rn_score_test],
            "NeuroSymb_score_Test": [ns_score_test],
            "Elapsed_Seconds": [elapsed],
            "Best_Loss": [best_loss],
            "Complexity": [best_complexity],
            "Problem_name": [problem_name]
        }

        if args.use_wandb:
            wandb.log(df_values)
            wandb_table = wandb.Table(
                dataframe=pysr_model.equations_[["equation", "score", "loss", "complexity"]]
            )
            wandb.log(
                {f"equations": wandb_table},
            )
        df = pd.DataFrame(df_values)
        df.to_csv(os.path.join(symbdir, "results.csv"), mode="w", header=True, index=False)
        send_full_report(df, logdir, pysr_model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()

    # args.data_size = 1000
    # args.iterations = 5
    # args.logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    # args.n_envs = 32
    # args.rounds = 300
    # args.binary_operators = ["+", "-", "greater"]
    # args.unary_operators = ["sin", "relu"]
    # args.denoise = True
    # args.wandb_tags = True
    # args.wandb_name = 8284
    # args.timeout_in_seconds = 36000
    # args.populations = 24
    # args.procs = 8
    # args.ncycles_per_iteration = 550
    print(args.bumper)

    if args.logdir is None:
        # Sparse coinrun:
        # args.logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"
        # # 10bn cartpole:
        args.logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
        print(f"No oracle provided.\nUsing Logdir: {args.logdir}")
    run_neurosymbolic_search(args)
