import os
import argparse
import re
from distutils.file_util import write_file

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

from cartpole.create_cartpole import create_cartpole, create_cartpole_env_pre_vec
from common.env.env_constructor import get_env_constructor

if os.name != "nt":
    from pysr import PySRRegressor

from helper_local import add_symbreg_args, get_config, DictToArgs, sigmoid, entropy_from_binary_prob, \
    get_actions_from_all, map_actions_to_values, concat_np_list, match
from symbolic_regression import run_neurosymbolic_search, load_nn_policy, generate_data  # , test_agent_mean_reward
from symbreg.agents import NeuralAgent, DeterministicNeuralAgent, CustomModel


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


def test_agent_mean_reward(agent, env, print_name, n=40, return_values=False):
    episodes = 0
    obs = env.reset()
    act = agent.forward(obs)
    cum_rew = np.zeros(len(act))
    episode_rewards = []
    episode_context = []
    while episodes < n:
        obs, rew, done, info = env.step(act)
        cum_rew += rew
        act = agent.forward(obs)
        if np.any(done):
            episodes += np.sum(done)
            episode_rewards += list(cum_rew[done])
            # TODO: make this general:
            episode_context += obs[done, (env.i_g - env.input_adjust):].tolist()
            cum_rew[done] = 0
    print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
    if return_values:
        return episode_rewards, episode_context
        # return {"mean":np.mean(episode_rewards), "std":np.std(episode_rewards)}
    return np.mean(episode_rewards)


def run_tests():
    dirs = [
        "logs/train/cartpole_swing/test/2024-05-01__14-19-53__seed_6033/symbreg/2024-05-13__10-27-13",
        # "logs/train/mountain_car/test/2024-05-03__15-46-58__seed_6033/symbreg/2024-05-14__00-45-59",
        # "logs/train/acrobot/test/2024-05-01__12-22-24__seed_6033/symbreg/2024-05-08__13-36-18",
        # "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033/symbreg/2024-05-03__11-14-03",
        # "logs/train/acrobot/test/2024-05-01__12-22-24__seed_6033/symbreg/2024-05-02__02-06-38",
        # "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033/symbreg/2024-05-02__13-37-11",

    ]
    for symbdir in dirs:
        try:
            test_saved_model(symbdir, n_envs=1000, n_rounds=10)#, override_model=CustomModel(degrees=16))
        except Exception as e:
            print(e)


def test_saved_model(symbdir, n_envs=10, n_rounds=10, override_model=None):
    # symbdir = "logs/train/acrobot/test/2024-05-01__12-22-24__seed_6033/symbreg/2024-05-02__02-06-38"
    # symbdir = "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033/symbreg/2024-05-02__12-03-40"
    # symbdir = "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033/symbreg/2024-05-02__13-37-11"
    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    logdir = re.search(r"(logs.*)symbreg", symbdir).group(1)
    pysr_model = PySRRegressor.from_file(pickle_filename)
    if override_model is not None:
        pysr_model = override_model
        name_suffix = "_cust"
    else:
        name_suffix = ""
    orig_cfg = get_config(logdir)
    env_name = orig_cfg["env_name"]
    cfg = get_config(symbdir)
    cfg["n_envs"] = n_envs
    cfg["rounds"] = n_rounds
    env_cons = get_env_constructor(orig_cfg["env_name"])
    args = DictToArgs(cfg)
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, args.n_envs)

    actions = get_actions_from_all(env)
    action_mapping = map_actions_to_values(actions)

    ns_agent = symbolic_agent_constructor(pysr_model, policy, args.stochastic, action_mapping)
    nn_agent = NeuralAgent(policy)

    train_params = env.get_params().copy()
    test_params = test_env.get_params().copy()

    if env_name == "cartpole" or env_name == "cartpole_swing":
        # test_params["max_gravity"] = 15
        # test_params["max_pole_length"] = 1.5
        # name_suffix = "_low"
        # test_params['min_gravity'] = 5
        # test_params['max_gravity'] = 9.8
        # test_params['min_cart_mass'] = 0.5
        # test_params['max_cart_mass'] = 1.0
        # test_params['min_pole_mass'] = 0.05
        # test_params['max_pole_mass'] = 0.1
        # test_params['min_pole_length'] = 0.25
        # test_params['max_pole_length'] = 0.5

        groups = ["gravity", "pole_length", "cart_mass", "pole_mass"]
        fancy_names = ["Gravity", "Pole Length", "Cart Mass", "Pole Mass"]
    elif env_name == "acrobot":
        # test_params["gravity"] = [5.0, 9.8]
        groups = ["gravity",
                  "link_length_1",
                  "link_length_2",
                  "link_mass_1",
                  "link_mass_2"]

        fancy_names = ["Gravity", "1st Link Length", "2nd Link Length", "1st Link Mass", "2nd Link Mass"]
    elif env_name == "mountain_car":
        groups = ["gravity"]
        fancy_names = ["Gravity"]
    else:
        raise NotImplementedError(f"Implement for env {env_name}")

    assert len(fancy_names) == len(groups), "Fancy names must have same length as groups"

    new_index = ["Train"] + fancy_names + ["All OOD"]

    ranges = {k: {} for k in groups}
    params = {}
    params["train"] = train_params
    for group in groups:
        temp_params = train_params.copy()
        keys = temp_params.keys()
        keys = [k for k in keys if re.search(group, k)]
        for k in keys:
            temp_params[k] = test_params[k]
            ranges[group][k] = {"train": train_params[k], "test": test_params[k]}
            # ranges[k] = {"train": train_params[k], "test": test_params[k]}
        params[group] = temp_params

    params["all"] = test_params

    if env_name in ["cartpole", "cartpole_swing", "mountain_car"]:
        train_ranges = {k: {re.sub(r"(min|max).*", r"train_\1", ks): vs["train"] for ks, vs in v.items()} for k, v in
                        ranges.items()}
        test_ranges = {k: {re.sub(r"(min|max).*", r"test_\1", ks): vs["test"] for ks, vs in v.items()} for k, v in
                       ranges.items()}
        tf = pd.DataFrame.from_dict(train_ranges)
        tf = tf._append(tf.from_dict(test_ranges))
    elif env_name == "acrobot":
        train_ranges = {k:
                            {f"train_min": v[k]["train"][0],
                             "train_max": v[k]["train"][1],
                             f"test_min": v[k]["test"][0],
                             "test_max": v[k]["test"][1]}

                        for k, v in ranges.items()
                        }
        tf = pd.DataFrame.from_dict(train_ranges)
    else:
        raise NotImplementedError(f"Implement for env {env_name}")

    # tf["all"] = np.nan
    # tf["train"] = np.nan

    results = {}
    record = {}
    for group, param in params.items():
        temp_env = env_cons(None, param, False)
        temp_env.reset(seed=6033)
        assert temp_env.get_params() == param, "params do not match"
        ns_reward, ns_context = test_agent_mean_reward(ns_agent, temp_env, f"NeuroSymb {group}", args.rounds, True)
        temp_env.reset(seed=6033)
        nn_reward, nn_context = test_agent_mean_reward(nn_agent, temp_env, f"Neural    {group}", args.rounds, True)
        results[group] = {"ns_mean": np.mean(ns_reward),
                          "ns_std": np.std(ns_reward),
                          "nn_mean": np.mean(nn_reward),
                          "nn_std": np.std(nn_reward),
                          "p_value": scipy.stats.ttest_ind(ns_reward, nn_reward, equal_var=False)[1]
                          }
        record[group] = {"ns": (ns_reward, ns_context), "nn": (nn_reward, nn_context)}

    df = pd.DataFrame.from_dict(results).T
    round_dict = {x: 1 for x in df.columns.values.tolist()}
    round_dict["p_value"] = 2
    df = df.round(round_dict)
    formatted_vals = {
        "Symbolic\nReward": concat_np_list([df["ns_mean"].values, " $\pm$ ", df["ns_std"].values], shape=(len(df),)),
        "Neural\nReward": concat_np_list([df["nn_mean"].values, " $\pm$ ", df["nn_std"].values], shape=(len(df),)),
        "P Value": df["p_value"].values.astype(str),
    }
    dff = pd.DataFrame.from_dict(formatted_vals)
    dff.index = new_index
    # df[df == "nan - nan"] = ""

    bolden_df(dff, df, greater_col="ns_mean", lesser_col="nn_mean", f_greater_col="Symbolic\nReward")
    bolden_df(dff, df, greater_col="nn_mean", lesser_col="ns_mean", f_greater_col="Neural\nReward")

    latex = dff.to_latex()  # os.path.join(symbdir, "results_table.tex"))

    formatted_vals = {
        "Train\nRange": concat_np_list([tf.T["train_min"].values, " - ", tf.T["train_max"].values], shape=(len(tf.T),)),
        "Test\nRange": concat_np_list([tf.T["test_min"].values, " - ", tf.T["test_max"].values], shape=(len(tf.T),)),
    }
    tff = pd.DataFrame.from_dict(formatted_vals)
    tff.index = fancy_names

    tf_latex = tff.to_latex()

    # dfo = pd.concat([tf.T, df], axis=1).reindex(["train"] + groups + ["all"])
    #
    # dfo.to_csv(os.path.join(symbdir, "results_table.csv"), index=True)
    #
    # # dfo = pd.read_csv(os.path.join(symbdir, "results_table.csv"), index_col=0)
    # formatted_vals = {
    #     "Train\nRange": concat_np_list([dfo["train_min"].values, " - ", dfo["train_max"].values], shape=(len(dfo),)),
    #     "Test\nRange": concat_np_list([dfo["test_min"].values, " - ", dfo["test_max"].values], shape=(len(dfo),)),
    #     "Symbolic\nReward": concat_np_list([dfo["ns_mean"].values, " $\pm$ ", dfo["ns_std"].values], shape=(len(dfo),)),
    #     "Neural\nReward": concat_np_list([dfo["nn_mean"].values, " $\pm$ ", dfo["nn_std"].values], shape=(len(dfo),)),
    #     "P Value": dfo["p_value"].values.astype(str),
    # }

    # caption = ("Environment Contextual Ranges for Training and Testing.")
    # metr_tab = wrap_latex_in_table(caption, f"{env_name}{name_suffix}-range-table", [tf_latex, latex])

    # assumes observation order = groups order
    obs_names = env.get_ob_names()

    latex_eqn = ns_agent.model.latex()
    # latex_eqn = re.sub(r"x_\{(\d)}",r"x\1", latex_eqn)
    #
    # for i, name in enumerate(obs_names):
    #     name = re.sub("_([^\d])",r"\_\1",name)
    #     latex_eqn = re.sub(f"x{i}", name, latex_eqn)

    formula = f"${latex_eqn}$"
    eq_dict = {}
    for i, name in enumerate(obs_names):
        name = re.sub("_([^\d])", r"\_\1", name)
        eq_dict[f"$x_{{{i}}}$"] = name

    dfk = pd.DataFrame.from_dict({"Key": eq_dict})
    form_latex = dfk.reindex(eq_dict.keys()).to_latex()

    caption = (f"{env_name}{name_suffix} - Training Environment uses all metrics from Train Range, "
               "while `All OOD' uses all metrics from Test Range. "
               "P Value is the probability that the means are different (T-test). "
               "Bold indicates the winner when P Value is less than 0.05.")
    label = f"{env_name}{name_suffix}-results-table"
    perf_tab = wrap_latex_in_table(caption, label, formula, [latex, tf_latex, form_latex])
    write_file(os.path.join(symbdir, f"{env_name}{name_suffix}_table.tex"), [perf_tab])

    if ns_agent.single_output:
        try:
            title = "Neural vs Symbolic Action Logits In and Out of Distribution"

            x_train, y_train, v_train = generate_data(ns_agent, env, 1000)
            x_test, y_test, v_test = generate_data(ns_agent, test_env, 1000)

            if not args.stochastic:
                y_test = action_mapping[y_test.argmax(1)]
                y_train = action_mapping[y_train.argmax(1)]

            l_train_ns = ns_agent.model.predict(x_train)
            l_test_ns = ns_agent.model.predict(x_test)
            min_x = min(np.min(y_test), np.min(y_train))
            max_x = max(np.max(y_test), np.max(y_train))
            min_y = min(np.min(l_test_ns), np.min(l_train_ns))
            max_y = max(np.max(l_test_ns), np.max(l_train_ns))
            plt.axline(xy1=(0, min_y), xy2=(0, max_y), color="black", linestyle="dashed", alpha=0.3)
            plt.axline(xy1=(min_x, 0), xy2=(max_x, 0), color="black", linestyle="dashed", alpha=0.3)
            plt.scatter(y_test, l_test_ns, label="Test", alpha=0.5)
            plt.scatter(y_train, l_train_ns, label="Train", alpha=0.5)
            plt.xlabel("Neural Logits")
            plt.ylabel("Symbolic Logits")
            plt.legend()
            plt.title(title)
            plt.savefig(os.path.join(symbdir, f"{env_name}{name_suffix}_logit.png"))
            plt.show()
        except Exception as e:
            pass

    ############################## graphs ###############################################

    title = "Reward Distributions"
    n_cols = 2
    n_row = int(np.ceil(len(record) / n_cols))
    fig, axes = plt.subplots(n_row, n_cols, figsize=(10, 10), sharex=True)
    for i, group in enumerate(["train"] + groups + ["all"]):
        ax = axes[i // n_cols, i % n_cols]
        group_vals = record[group]
        nn = group_vals["nn"][0]
        ns = group_vals["ns"][0]
        ax.hist(nn, label="Neural", alpha=.75)
        ax.hist(ns, label="Symbolic", alpha=.75)
        ax.axvline(np.mean(nn), label="Neural Mean", color="blue")
        ax.axvline(np.mean(ns), label="Symbolic Mean", color="red")
        # ax.legend(loc="upper left")
        ax.set_title(new_index[i])
    # plt.legend()
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, "upper left")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(symbdir, f"{env_name}{name_suffix}_hist.png"))

    plt.show()

    title = "Min reward analysis"
    rew = np.array(record["all"]["ns"][0])
    obs = np.array(record["all"]["ns"][1])
    flt = rew == np.min(rew)
    n_cols = 2
    n_row = int(np.ceil(len(fancy_names) / n_cols))
    fig, axes = plt.subplots(n_row, n_cols, figsize=(10, 10), sharex=False)
    for i, ob in enumerate(obs[flt].T):
        if i < len(fancy_names):
            if len(axes.shape) == 1:
                ax = axes[i]
            else:
                ax = axes[i // n_cols, i % n_cols]
            ax.hist(obs[flt == False].T[i], label="Non-min Reward", alpha=.75)
            ax.hist(ob, label="Min Reward", alpha=.75)
            ax.set_title(fancy_names[i])
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, "upper left")
    fig.suptitle(title)
    plt.show()

    title = "Reward vs Context Value"
    obs_order = groups
    n_cols = 2
    n_row = int(np.ceil(len(obs_order) / n_cols))
    fig, axes = plt.subplots(n_row, n_cols, figsize=(10, 10), sharex=False)
    for i, group in enumerate(obs_order):
        if len(axes.shape) == 1:
            ax = axes[i]
        else:
            ax = axes[i // n_cols, i % n_cols]
        group_vals = record[group]
        ns = group_vals["nn"]
        obs = np.array(ns[1])
        x = obs[:, i].squeeze()
        y = np.array(ns[0])
        # flt = y!=500
        flt = np.ones_like(y).astype(bool)
        ax.scatter(
            x=x[flt],
            y=y[flt],
            label="Neural", alpha=.5
            # c=[i for i in range(np.sum(flt))]
        )

        nn = group_vals["ns"]
        obs = np.array(nn[1])
        x = obs[:, i].squeeze()
        y = np.array(nn[0])
        # flt = y!=500
        flt = np.ones_like(y).astype(bool)
        ax.scatter(
            x=x[flt],
            y=y[flt],
            label="Symbolic", alpha=.5
            # c=[i for i in range(np.sum(flt))]
        )
        # ax.hist(record[group]["nn"][0])
        ax.set_title(fancy_names[i])
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, "upper left")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(symbdir, f"{env_name}{name_suffix}_scatter1.png"))
    plt.show()

    for metric in obs_order:
        title = f"Reward vs Context Value Within {metric} env."  # where grav>24"
        n_cols = 2
        n_row = int(np.ceil(len(obs_order) / n_cols))
        fig, axes = plt.subplots(n_row, n_cols, figsize=(10, 10), sharex=False)
        for i, group in enumerate(obs_order):
            if len(axes.shape) == 1:
                ax = axes[i]
            else:
                ax = axes[i // n_cols, i % n_cols]
            group_vals = record[metric]
            group_index = match(np.array([metric]), np.array(obs_order))
            ns = group_vals["nn"]
            obs = np.array(ns[1])
            x = obs[:, i].squeeze()
            y = np.array(ns[0])
            # flt = y!=500
            flt = np.ones_like(y).astype(bool)
            # flt = y==-500
            # flt = (obs[:,group_index]>24).squeeze()
            ax.scatter(
                x=x[flt],
                y=y[flt],
                label="Neural", alpha=.5
                # c=[i for i in range(np.sum(flt))]
            )

            nn = group_vals["ns"]
            obs = np.array(nn[1])
            x = obs[:, i].squeeze()
            y = np.array(nn[0])
            # flt = y!=500
            flt = np.ones_like(y).astype(bool)
            # flt = y==-500
            # flt = (obs[:,group_index]>24).squeeze()
            ax.scatter(
                x=x[flt],
                y=y[flt],
                label="Symbolic", alpha=.5
                # c=[i for i in range(np.sum(flt))]
            )
            # ax.hist(record[group]["nn"][0])
            ax.set_title(fancy_names[i])
        lines_labels = [ax.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, "upper left")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        # plt.savefig(os.path.join(symbdir, f"{env_name}{name_suffix}_scatter3.png"))
        plt.show()

    fig, axes = plt.subplots(n_row, n_cols, figsize=(10, 10), sharex=False)
    for i, group in enumerate(obs_order):
        if len(axes.shape) == 1:
            ax = axes[i]
        else:
            ax = axes[i // n_cols, i % n_cols]
        obs = np.array(record["all"]["nn"][1])
        x = obs[:, i].squeeze()
        y = np.array(record["all"]["nn"][0])
        # flt = y!=500
        flt = np.ones_like(y).astype(bool)
        ax.scatter(
            x=x[flt],
            y=y[flt],
            c=[i for i in range(np.sum(flt))]
        )
        ax.set_title(group)
    plt.savefig(os.path.join(symbdir, f"{env_name}{name_suffix}_scatter2.png"))
    plt.show()

    return None


def bolden_df(df,
              dfo,
              greater_col="ns_mean",
              lesser_col="nn_mean",
              f_greater_col="Symbolic\nReward"
              ):
    symb_win = np.bitwise_and(dfo["p_value"] < 0.05, dfo[greater_col] > dfo[lesser_col]).values
    vals = df[f_greater_col].loc[symb_win]

    df[f_greater_col].loc[symb_win] = concat_np_list(["\\textbf{", vals.values, "}"], (len(vals),))


def wrap_latex_in_table(caption, label, formula, latex):
    tabulars = '\n'.join(latex)
    return f"\\begin{{table}}\n\\caption{{{caption}\\\\\\\\\n\\textbf{{Symbolic Formula: {formula}}}}}\n\\label{{{label}}}\n\\centering\n{tabulars}\n\\end{{table}}"


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
    args.logdir = "logs/train/mountain_car/test/2024-05-03__15-46-58__seed_6033"
    # args.logdir = "logs/train/cartpole_swing/test/2024-05-01__14-19-53__seed_6033"
    args.n_envs = 100
    args.rounds_per_epoch = 300
    args.binary_operators = ["+", "-", "*", "greater", "/"]
    args.unary_operators = ["sin", "relu", "log", "exp", "sign", "sqrt", "square"]

    args.denoise = False
    args.use_wandb = True
    args.wandb_tags = ["test"]
    args.weight_metric = "entropy"
    args.wandb_name = "manual"
    # args.populations = 24
    args.model_selection = "best"
    args.ncycles_per_iteration = 2000
    args.bumper = True
    args.loss_function = "capped_sigmoid"
    for stoch in [True]:
        args.stochastic = stoch
        run_neurosymbolic_search(args)


if __name__ == "__main__":
    run_tests()
    # format_results()
    # test_saved_model()
    # test_agent_specific_environment()
    # run_saved_model()
    # run_deterministic_agent()
    # run_symb_reg_local()
