import os
import argparse
import re
from distutils.file_util import write_file

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cartpole.create_cartpole import create_cartpole, create_cartpole_env_pre_vec
from common.env.env_constructor import get_env_constructor

if os.name != "nt":
    from pysr import PySRRegressor

from helper_local import add_symbreg_args, get_config, DictToArgs, sigmoid, entropy_from_binary_prob, \
    get_actions_from_all, map_actions_to_values, concat_np_list
from symbolic_regression import run_neurosymbolic_search, load_nn_policy, generate_data, test_agent_mean_reward
from symbreg.agents import NeuralAgent, DeterministicNeuralAgent


def run_deterministic_agent():
    # logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    # logdir = "logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"
    logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"
    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, 32)

    nn_agent = NeuralAgent(policy)
    nn_score_train = test_agent(nn_agent, env, "Neural Train", 300)
    nn_score_test = test_agent(nn_agent, test_env, "Neural Test", 300)

    dn_agent = DeterministicNeuralAgent(policy)
    dn_score_train = test_agent(dn_agent, env, "DetNeural Train", 300)
    dn_score_test = test_agent(dn_agent, test_env, "DetNeural Test", 300)


def test_agent_specific_environment():
    # Here we test different planet gravities
    n_envs = 32
    logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    symbdir = os.path.join(logdir, "symbreg/2024-04-12__17-38-41/")
    # symbdir = os.path.join(logdir, "symbreg/2024-04-11__15-12-07/")
    # logdir = "logs/train/cartpole/cartpole/2024-04-15__15-45-47__seed_6033"
    # symbdir = os.path.join(logdir, "symbreg/2024-04-15__16-36-44/")
    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    pysr_model = PySRRegressor.from_file(pickle_filename)

    cfg = get_config(symbdir)
    args = DictToArgs(cfg)

    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, n_envs)

    nn_agent = NeuralAgent(policy)
    ns_agent = symbolic_agent_constructor(pysr_model, policy, args.stochastic, None)
    nn_scores = []
    ns_scores = []
    gs = [9.8, 12, 14, 16, 18, 20, 22, 24, 26]
    for gravity in gs:
        env_args = {"n_envs": n_envs,
                    "env_name": "CartPole-v1",
                    "degrees": 12,
                    "h_range": 2.4,
                    "gravity": gravity,  # 11.15,#10.44,#3.7,#24.8
                    }
        mars_test_env = create_cartpole_env_pre_vec(env_args, render=False, normalize_rew=False)

        rounds = 100
        # nn_score_train = test_agent(nn_agent, env, "Neural Train", rounds)
        # nn_score_test = test_agent(nn_agent, test_env, "Neural Test", rounds)
        nn_score_mars = test_agent(nn_agent, mars_test_env, f"Neural\t{gravity}", rounds)
        nn_scores.append(nn_score_mars)
        # ns_score_train = test_agent(ns_agent, env, "NeuroSymb Train", rounds)
        # ns_score_test = test_agent(ns_agent, test_env, "NeuroSymb Test", rounds)
        ns_score_mars = test_agent(ns_agent, mars_test_env, f"NeuroSymb\t{gravity}", rounds)
        ns_scores.append(ns_score_mars)

    plt.plot(gs, nn_scores, label="Neural")
    plt.plot(gs, ns_scores, label="NeuroSymb")
    plt.legend()
    plt.show()
    print("done")


def test_saved_model():
    symbdir = "logs/train/acrobot/test/2024-04-25__10-03-20__seed_6033/symbreg/2024-04-25__16-54-37/"
    symbdir = "logs/train/cartpole/test/2024-04-26__12-37-41__seed_40/symbreg/2024-04-27__16-00-09"
    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    logdir = re.search(r"(logs.*)symbreg", symbdir).group(1)
    pysr_model = PySRRegressor.from_file(pickle_filename)
    orig_cfg = get_config(logdir)
    cfg = get_config(symbdir)
    cfg["n_envs"] = 128
    cfg["rounds"] = 300
    env_cons = get_env_constructor(orig_cfg["env_name"])
    args = DictToArgs(cfg)
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, args.n_envs)

    actions = get_actions_from_all(env)
    action_mapping = map_actions_to_values(actions)

    ns_agent = symbolic_agent_constructor(pysr_model, policy, args.stochastic, action_mapping)
    nn_agent = NeuralAgent(policy)

    train_params = env.get_params().copy()
    test_params = test_env.get_params().copy()

    groups = ["gravity", "pole_length", "cart_mass", "pole_mass"]
    ranges = {k: {} for k in groups}
    params = {}
    for group in groups:
        temp_params = train_params.copy()
        keys = temp_params.keys()
        keys = [k for k in keys if re.search(group, k)]
        for k in keys:
            temp_params[k] = test_params[k]
            ranges[group][k] = {"train": train_params[k], "test": test_params[k]}
            # ranges[k] = {"train": train_params[k], "test": test_params[k]}
        params[group] = temp_params

    params["train"] = train_params
    params["all"] = test_params

    # TODO: this will have to be different for acrobot:
    train_ranges = {k: {re.sub(r"(min|max).*", r"train_\1", ks): vs["train"] for ks, vs in v.items()} for k, v in
                    ranges.items()}
    test_ranges = {k: {re.sub(r"(min|max).*", r"test_\1", ks): vs["test"] for ks, vs in v.items()} for k, v in
                   ranges.items()}

    tf = pd.DataFrame.from_dict(train_ranges)
    tf = tf._append(tf.from_dict(test_ranges))
    tf["train"] = np.nan
    tf["all"] = np.nan

    results = {}
    for group, param in params.items():
        temp_env = env_cons(None, param, False)
        assert temp_env.get_params() == param, "params do not match"
        ns_score = test_agent_mean_reward(ns_agent, temp_env, f"NeuroSymb {group}", args.rounds, True)
        nn_score = test_agent_mean_reward(nn_agent, temp_env, f"Neural    {group}", args.rounds, True)
        results[group] = {"ns_mean": ns_score["mean"],
                          "ns_std": ns_score["std"],
                          "nn_mean": nn_score["mean"],
                          "nn_std": nn_score["std"],
                          }

    df = pd.DataFrame.from_dict(results).T
    df = df.round(1)
    dfo = pd.concat([tf.T, df], axis=1).reindex(["train"] + groups + ["all"])

    dfo.to_csv(os.path.join(symbdir, "results_table.csv"), index=True)

    env_name = orig_cfg["cartpole"]

    # dfo = pd.read_csv(os.path.join(symbdir, "results_table.csv"), index_col=0)
    formatted_vals = {
        "Train\nRange": concat_np_list([dfo["train_min"].values, " - ", dfo["train_max"].values], shape=(len(dfo),)),
        "Test\nRange": concat_np_list([dfo["test_min"].values, " - ", dfo["test_max"].values], shape=(len(dfo),)),
        "Symbolic\nReward": concat_np_list([dfo["ns_mean"].values, " $\pm$ ", dfo["ns_std"].values], shape=(len(dfo),)),
        "Neural\nReward": concat_np_list([dfo["nn_mean"].values, " $\pm$ ", dfo["nn_std"].values], shape=(len(dfo),)),
    }
    new_index = ["Train", "Gravity", "Pole Length", "Cart Mass", "Pole Mass", "All OOD"]

    df = pd.DataFrame.from_dict(formatted_vals)
    df.index = new_index
    df[df == "nan - nan"] = ""

    latex = df.to_latex()  # os.path.join(symbdir, "results_table.tex"))

    caption = "Training Environment uses all metrics from Train Range, while `All OOD' uses all metrics from Test Range."
    label = f"{env_name}-results-table"
    output_str = wrap_latex_in_table(caption, label, latex)
    write_file(os.path.join(symbdir, f"{env_name}_table.tex"), [output_str])

    return None


def format_results():
    symbdir = "logs/train/cartpole/test/2024-04-26__12-37-41__seed_40/symbreg/2024-04-27__16-00-09"

    return latex

    return None


def wrap_latex_in_table(caption, label, latex):
    return f"\\begin{{table}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n{latex}\n\\end{{table}}"


def run_saved_model():
    # data_size = 1000
    # n_envs = 32
    # rounds = 300

    # logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    # symbdir = os.path.join(logdir, "symbreg/2024-04-12__17-38-41/")
    # symbdir = os.path.join(logdir, "symbreg/2024-04-11__15-12-07/")
    # logdir = "logs/train/cartpole/cartpole/2024-04-15__15-45-47__seed_6033"
    # symbdir = os.path.join(logdir, "symbreg/2024-04-15__16-36-44/")
    logdir = "logs/train/acrobot/test/2024-04-25__10-03-20__seed_6033"
    symbdir = os.path.join(logdir, "symbreg/2024-04-25__16-53-11/")

    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    pysr_model = PySRRegressor.from_file(pickle_filename)

    cfg = get_config(symbdir)
    args = DictToArgs(cfg)

    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, args.n_envs)

    actions = get_actions_from_all(env)
    action_mapping = map_actions_to_values(actions)

    ns_agent = symbolic_agent_constructor(pysr_model, policy, args.stochastic, action_mapping)
    # X, Y, V = generate_data(ns_agent, env, int(1000))
    # nn_agent = NeuralAgent(policy)
    ns_score_train = test_agent_mean_reward(ns_agent, env, "NeuroSymb Train", args.rounds)
    # nn_score_train = test_agent(nn_agent, env, "Neural    Train", args.rounds)
    return
    Y, Y_hat = plot_action_entropy_vs_pole_angle(X, Y, args, policy, pysr_model, sampler, symbdir)

    plt.scatter(sigmoid(Y), sigmoid(Y_hat), c=V)
    plt.scatter(sigmoid(Y), sigmoid(Y))
    plt.show()

    plt.scatter(Y, Y_hat, c=V)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend(['Value'])
    plt.show()

    p = sigmoid(Y)
    Y_act = np.int32(np.random.random(p.shape) < p)

    p_hat = sigmoid(Y_hat)
    Y_hat_act = np.int32(np.random.random(p_hat.shape) < p)

    print(f"same action:{np.sum(Y_act == Y_hat_act) / len(Y_act)}")

    plt.scatter(Y_act - Y_hat_act, V)
    plt.show()
    plt.hist(V[Y_act == Y_hat_act], bins=50)
    plt.hist(V[Y_act != Y_hat_act], bins=50)
    plt.show()

    plt.scatter(Y_hat, V, c=Y_hat_act)
    plt.show()

    plt.scatter(Y, V, c=Y_act)
    plt.show()

    print("done")


def plot_action_entropy_vs_pole_angle(X, Y, args, policy, pysr_model, sampler, symbdir):
    X[:, 2] = (np.random.rand(X.shape[0]) - .5) * 28 / 360 * 2 * np.pi
    Y_hat = pysr_model.predict(X)
    _, Y, _, _ = sampler(policy, X, args.stochastic)
    p_hat = sigmoid(Y_hat)
    p = sigmoid(Y)
    ent_hat = entropy_from_binary_prob(p_hat)
    ent = entropy_from_binary_prob(p)
    pole_degrees = X[:, 2] * 360 / (2 * np.pi)
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].scatter(x=pole_degrees, y=ent)
    ax[0].set_title('Neural')
    ax[1].scatter(x=pole_degrees, y=ent_hat)
    ax[1].set_title('Symbolic')

    for _ax in ax:
        _ax.vlines([9, -9], 0, max(np.nanmax(ent), np.nanmax(ent_hat)), linestyles='dashed')

    fig.text(0.5, 0.04, 'Pole Angle Degrees', ha='center')
    fig.text(0.04, 0.5, 'Action Entropy', va='center', rotation='vertical')
    plt.savefig(os.path.join(symbdir, "pole_angle_entropy.png"))
    plt.show()
    return Y, Y_hat


def run_symb_reg_local():
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()

    args.data_size = 100
    args.iterations = 1
    # args.logdir = "logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"
    # args.logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    # args.logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"
    # args.logdir = "logs/train/acrobot/test/2024-04-25__10-03-20__seed_6033"
    args.logdir = "logs/train/cartpole/test/2024-04-26__12-37-41__seed_40"
    args.n_envs = 128
    args.rounds = 300
    args.binary_operators = ["+", "-", "*", "greater", "/"]
    args.unary_operators = ["sin", "relu", "log", "exp", "sign", "sqrt", "square"]

    args.denoise = False
    args.use_wandb = True
    args.wandb_tags = ["test"]
    args.weight_metric = "value"
    args.wandb_name = "manual"
    # args.populations = 24
    args.model_selection = "best"
    args.ncycles_per_iteration = 2000
    args.bumper = True
    args.loss_function = "capped_sigmoid"
    for stoch in [True, False]:
        args.stochastic = stoch
        run_neurosymbolic_search(args)


if __name__ == "__main__":
    format_results()
    # test_saved_model()
    # test_agent_specific_environment()
    # run_saved_model()
    # run_deterministic_agent()
    # run_symb_reg_local()
