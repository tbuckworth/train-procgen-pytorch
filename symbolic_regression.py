import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import sympy
from torch import cuda

import wandb
from discrete_env.mountain_car_pre_vec import create_mountain_car

from email_results import send_images_first_last
from symbreg.agents import SymbolicAgent, NeuralAgent, RandomAgent, NeuroSymbolicAgent, GraphSymbolicAgent, \
    DoubleGraphSymbolicAgent, PureGraphAgent

if os.name != "nt":
    from pysr import PySRRegressor
# Important! keep torch after pysr
import torch
from common.env.procgen_wrappers import create_env, create_procgen_env
from helper_local import get_config, get_path, balanced_reward, load_storage_and_policy, \
    load_hparams_for_model, floats_to_dp, dict_to_html_table, wandb_login, add_symbreg_args, DictToArgs, \
    inverse_sigmoid, sigmoid, sample_from_sigmoid, map_actions_to_values, get_actions_from_all, \
    entropy_from_binary_prob, get_saved_hyperparams, softmax, sample_numpy_probs
from common.env.env_constructor import get_env_constructor
from cartpole.create_cartpole import create_cartpole
from boxworld.create_box_world import create_bw_env
from matplotlib import pyplot as plt

# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\PycharmProjects\train-procgen-pytorch\venv\julia_env\pyjuliapkg\install\bin"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\AppData\Local\Microsoft\WindowsApps"
# os.environ["PYTHON_JULIACALL_BINDIR"] = r"C:\Users\titus\.julia\juliaup\julia-1.10.0+0.x64.w64.mingw32\bin"

pysr_loss_functions = {
    "sigmoid": "SigmoidLoss()",
    "exp": "ExpLoss()",
    "l2marg": "L2MarginLoss()",
    "logitmarg": "LogitMarginLoss()",
    "perceptron": "PerceptronLoss()",
    "logitdist": "LogitDistLoss()",
    "mse": "loss(prediction, target) = (prediction - target)^2",
    "mce": "loss(prediction, target) = abs(prediction - target)^3",
    "capped_sigmoid": "loss(y_hat, y) = 1 - tanh(y*y_hat) + abs(y_hat-y)",
}


def find_model(X, Y, symbdir, save_file, weights, args):
    model = PySRRegressor(
        equation_file=get_path(symbdir, save_file),
        niterations=args.iterations,  # < Increase me for better results
        binary_operators=args.binary_operators,
        unary_operators=args.unary_operators,
        weights=weights,
        denoise=args.denoise,
        extra_sympy_mappings={"greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True)),
                              "bal1": lambda x2, x3: x2+2*x3,
                              "bal2": lambda x1, x2: sympy.Piecewise((1.0, x1 + x2 > -0.165), (0.0, True)),
                              },

        # "balance": lambda x1, x2, x3, x7: sympy.Piecewise((1.0, x1 + (x2+2*x3)/x7 > -0.165), (0.0, True))},
        elementwise_loss=pysr_loss_functions[args.loss_function],
        timeout_in_seconds=args.timeout_in_seconds,
        populations=args.populations,
        procs=args.procs,
        batching=args.data_size > 1000,
        weight_optimize=0.001 if args.ncycles_per_iteration > 550 else 0.0,
        ncycles_per_iteration=args.ncycles_per_iteration,
        bumper=args.bumper,
        model_selection=args.model_selection,
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


def load_nn_policy(logdir, n_envs=2, render=False):
    cfg = get_config(logdir)
    cfg["n_envs"] = n_envs
    env_name = cfg["env_name"]
    create_venv = get_env_constructor(env_name)
    symbolic_agent_constructor = SymbolicAgent

    if env_name in ["coinrun", "boxworld"]:
        symbolic_agent_constructor = NeuroSymbolicAgent

    if cfg["architecture"] == "graph-transition":
        symbolic_agent_constructor = GraphSymbolicAgent

    if cfg['architecture'] == "double-graph":
        symbolic_agent_constructor = DoubleGraphSymbolicAgent

    if cfg['architecture'] == 'graph':
        symbolic_agent_constructor = PureGraphAgent

    device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")
    hyperparameters, last_model = load_hparams_for_model(cfg["param_name"], logdir, n_envs)
    hyperparameters["n_envs"] = n_envs

    if render:
        cfg["render"] = True
    tmp_args = DictToArgs(cfg)
    # This seems to be necessary as training from scratch produces a different result than using a trained model:
    hyperparameters["normalize_rew"] = False
    env = create_venv(tmp_args, hyperparameters, is_valid=False)
    test_env = create_venv(tmp_args, hyperparameters, is_valid=True)

    action_names, done, hidden_state, obs, policy, storage = load_storage_and_policy(device, env, hyperparameters,
                                                                                     last_model, logdir, n_envs)

    return policy, env, symbolic_agent_constructor, test_env


def drop_first_dim(arr):
    shp = np.array(arr.shape)
    new_shape = tuple(np.concatenate(([np.prod(shp[:2])], shp[2:])))
    return arr.reshape(new_shape)


def generate_data(agent, env, n):
    observation = env.reset()
    x, y, act, value = agent.sample(observation)
    X = x
    Y = y
    V = value
    while len(X) < n:
        observation, rew, done, info = env.step(act)
        x, y, act, v = agent.sample(observation)
        X = np.append(X, x, axis=0)
        Y = np.append(Y, y, axis=0)
        V = np.append(V, v, axis=0)
    return X, Y, V


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


def test_agent_mean_reward(agent, env, print_name, n=40, return_values=False):
    print(print_name)
    episodes = 0
    obs = env.reset()
    act = agent.forward(obs)
    cum_rew = np.zeros(len(act))
    episode_rewards = []
    while episodes < n:
        obs, rew, done, info = env.step(act)
        cum_rew += rew
        act = agent.forward(obs)
        if np.any(done):
            episodes += np.sum(done)
            episode_rewards += list(cum_rew[done])
            cum_rew[done] = 0
    print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
    if return_values:
        return episode_rewards
        # return {"mean":np.mean(episode_rewards), "std":np.std(episode_rewards)}
    return np.mean(episode_rewards)


# def sample_policy_with_symb_model(model, policy, observation):
#     with torch.no_grad():
#         obs = torch.FloatTensor(observation).to(policy.device)
#         x = policy.embedder.forward_to_pool(obs)
#         h = model(x)
#         dist, value = policy.hidden_to_output(h)
#         # y = dist.logits.detach().cpu().numpy()
#         act = dist.sample()
#     return act.cpu().numpy()


def get_coinrun_test_env(logdir, n_envs):
    cfg = get_config(logdir)
    hyperparameters = get_saved_hyperparams(logdir)
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


def send_full_report(df, logdir, symbdir, model, args, dfs):
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
    roll_window = 5
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

    hline_dict = {"NeuroSymb Train": [ns_train, "blue", "dashed"],
                  "NeuroSymb Test": [ns_test, "orange", "dashed"],
                  "Random Train": [rn_train, "blue", "dotted"],
                  "Random Test": [rn_test, "orange", "dotted"], }
    for key in hline_dict.keys():
        plt_hline(dfl2, key, hline_dict[key][0], hline_dict[key][1], hline_dict[key][2])

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

    plot_file = os.path.join(symbdir, "training_plot.png")
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

    test_improved = (ns_test > nn_test) and (ns_test > rn_test)
    train_improved = ns_train > nn_train and (ns_train > rn_train)

    statement = "Failed"
    if test_improved and train_improved:
        statement = "Complete Success"
    if not test_improved and train_improved:
        statement = "Improved Training Reward"
    if test_improved and not train_improved:
        statement = "Improved Generalization"

    lims = [
        min(dfs.logit.min(), dfs.logit_estimate.min()),
        max(dfs.logit.max(), dfs.logit_estimate.max())
    ]
    dfs.plot.scatter(x="logit", y="logit_estimate", c="value")
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    scatter_file = os.path.join(symbdir, "training_scatter.png")
    plt.savefig(scatter_file)

    body_text = f"<br><b>{statement}</b><br>{tab_code}<br><b>Learned Formula:</b><br><p>{eqn_str}</p><br>{params}"
    send_images_first_last([plot_file, scatter_file], "PySR Results", body_text=body_text)
    print("")


def split_df_by_index_and_pivot(df):
    dfv = df.filter(regex="_score_").T
    dfv2 = pd.Series(dfv.index).str.split("_score_", expand=True)
    dfv2.columns = ["Model", "Environment"]
    dfv2["mean_reward"] = dfv[0].values.round(decimals=2)
    dfw = dfv2.pivot(columns="Environment", index="Model", values="mean_reward")
    tab_code = dfw.to_html(index=True)


def plt_hline(dfl2, label, ns_train, colour, style):
    plt.hlines(y=ns_train,
               xmin=0,
               xmax=dfl2.index.max(),
               label=label,
               linestyles=style,
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


def get_entropy(Y):
    # only legit for single action (cartpole)
    if len(Y.shape) == 1:
        p = sigmoid(Y)
        q = 1 - p
        return -p * np.log(p) - q * np.log(q)
    ents = -(np.exp(Y) * Y).sum(-1)
    return ents


def one_hot(targets, nb_classes):
    return np.eye(nb_classes)[targets]


def run_neurosymbolic_search(args):
    data_size = args.data_size
    logdir = args.logdir
    n_envs = args.n_envs
    if n_envs < 2:
        raise Exception("n_envs must be at least 2")
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
        cfg["symbdir"] = symbdir
        if args.wandb_group is not None:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name, group=args.wandb_group)
        else:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name)

    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs)
    # policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, n_envs)
    ns_agent = symbolic_agent_constructor(None, policy, args.stochastic, None)
    X, Y, V = generate_data(ns_agent, env, int(data_size))
    e = get_entropy(Y)

    try:
        actions = get_actions_from_all(env)
    except NotImplementedError:
        actions = np.array([f"action_{i}" for i in range(env.action_space.n)])
    action_mapping = None
    if not args.stochastic:
        save_Y = Y.copy()
        # What if actions aren't implemented? improve map_actions...
        action_mapping = map_actions_to_values(actions)
        # map actions to 0:2pi
        Y = action_mapping[Y.argmax(1)]

    print("data generated")
    if os.name != "nt":
        weights = None
        if args.weight_metric is not None:
            if args.weight_metric == "entropy":
                weights = 1/e
                weights[np.isinf(weights)] = weights[np.isinf(weights)==False].max()
            elif args.weight_metric == "value":
                weights = V
        pysr_model, elapsed = find_model(X, Y, symbdir, save_file, weights, args)
        ns_agent = symbolic_agent_constructor(pysr_model, policy, args.stochastic, action_mapping)
        nn_agent = NeuralAgent(policy)
        rn_agent = RandomAgent(env.action_space.n)

        ns_score_train = test_agent_mean_reward(ns_agent, env, "NeuroSymb Train", rounds)
        nn_score_train = test_agent_mean_reward(nn_agent, env, "Neural    Train", rounds)
        rn_score_train = test_agent_mean_reward(rn_agent, env, "Random    Train", rounds)

        ns_score_test = test_agent_mean_reward(ns_agent, test_env, "NeuroSymb  Test", rounds)
        nn_score_test = test_agent_mean_reward(nn_agent, test_env, "Neural     Test", rounds)
        rn_score_test = test_agent_mean_reward(rn_agent, test_env, "Random     Test", rounds)

        best = pysr_model.get_best()
        if type(best) != list:
            best = [best]
        best_loss = np.mean([x.loss for x in best])
        best_complexity = np.mean([x.complexity for x in best])
        problem_name = re.search("logs/train/([^/]*)/", logdir).group(1)

        df_values = {
            "Random_score_Train": [rn_score_train],
            "Neural_score_Train": [nn_score_train],
            "NeuroSymb_score_Train": [ns_score_train],
            "Neural_score_Test": [nn_score_test],
            "Random_score_Test": [rn_score_test],
            "NeuroSymb_score_Test": [ns_score_test],
            "Elapsed_Seconds": [elapsed],
            "Mean_Best_Loss": [best_loss],
            "Mean_Complexity_of_Best": [best_complexity],
            "Problem_name": [problem_name]
        }

        Y_hat = pysr_model.predict(X)
        # TODO: Use agent methods to clean this all up:
        if ns_agent.single_output:
            if args.stochastic:
                p = sigmoid(Y)
                Y_act = sample_from_sigmoid(p)
                p_hat = sigmoid(Y_hat)
                Y_hat_act = sample_from_sigmoid(p)
                ent = entropy_from_binary_prob(p)
                ent_hat = entropy_from_binary_prob(p_hat)
            else:
                Y_act = Y
                p = Y
                p_hat = Y_hat
                Y_hat_act = Y_hat
                ent = np.zeros_like(p)
                ent_hat = np.zeros_like(p_hat)
        else:
            if not args.stochastic:
                p = softmax(save_Y)
                ent = -(p * np.log(p)).sum(1)
                Y_act = sample_numpy_probs(p)
                p = p[np.arange(len(p)), Y_act]
                Y_hat_act = ns_agent.pred_to_action(Y_hat)
                p_hat = np.ones_like(Y)  # one_hot(Y_hat_act, save_Y.shape[-1])
                ent_hat = np.zeros_like(Y_hat)
            else:
                p = softmax(Y)
                p_hat = softmax(Y_hat)
                Y_hat_act = sample_numpy_probs(p_hat)
                ent_hat = -(p_hat * np.log(p_hat)).sum(1)
                Y_act = sample_numpy_probs(p)
                ent = -(p * np.log(p)).sum(1)

        df_values["Entropy_Pred"] = [ent_hat.mean()]
        df_values["Entropy"] = [ent.mean()]

        shp = Y.shape
        if len(shp) == 1:
            shp = (shp[0], 1)

        if not args.stochastic:
            action_vector = actions[Y_act]
        elif len(actions) == 2:
            action_vector = np.repeat(actions[-1], shp[0])
        else:
            action_vector = np.repeat(actions, shp[0])

        all_metrics = np.vstack(
            (action_vector,
             Y.reshape(np.prod(shp)),
             np.tile(V, shp[-1]),
             Y_hat.reshape(np.prod(shp)),
             p.reshape(np.prod(shp)),
             p_hat.reshape(np.prod(shp)),
             np.tile(Y_act, shp[-1]),
             np.tile(Y_hat_act, shp[-1]),
             np.tile(ent, shp[-1]),
             np.tile(ent_hat, shp[-1])
             )
        ).T
        # all_metrics = np.vstack((Y, V, Y_hat, p, p_hat, Y_act, Y_hat_act)).T
        columns = ["action", "logit", "value", "logit_estimate", "prob", "prob_estimate",
                   "sampled_action", "sampled_action_estimate", "entropy", "entropy_estimate"]
        if problem_name == "cartpole":
            all_metrics = np.hstack((X, all_metrics))
            state_features = [
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity",
                "gravity",
                "pole_length",
                "cart_mass",
                "pole_mass",
                "force_magnitude",
            ]
            columns = state_features + columns
        dfs = pd.DataFrame(all_metrics, columns=columns)
        dfs.loc[:, dfs.columns != "action"] = dfs.loc[:, dfs.columns != "action"].astype(float)

        if args.use_wandb:
            wandb.log({k: df_values[k][0] for k in df_values.keys()})
            # wandb_table = wandb.Table(
            #     # make this work for multiple equations:
            #     dataframe=pysr_model.equations_[["equation", "score", "loss", "complexity"]]
            # )
            wandb_metrics = wandb.Table(dataframe=dfs)
            wandb.log(
                # {#f"equations": wandb_table,
                {f"metrics": wandb_metrics},
            )

        df = pd.DataFrame(df_values)
        df.to_csv(os.path.join(symbdir, "results.csv"), mode="w", header=True, index=False)
        send_full_report(df, logdir, symbdir, pysr_model, args, dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()

    print(f"\nBinary Ops:\n{' '.join(args.binary_operators)}\n")

    if args.logdir is None:
        raise Exception("No oracle provided. Please provide logdir argument")
        # Sparse coinrun:
        # args.logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"

        # 10bn cartpole:
        # args.logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"

        # # Sparse Boxworld, overfit:
        # args.logdir = "logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"
        print(f"No oracle provided.\nUsing Logdir: {args.logdir}")
    run_neurosymbolic_search(args)
