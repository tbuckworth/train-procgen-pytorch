import argparse
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import wandb
from gp import bayesian_optimisation
from helper_local import wandb_login


def get_wandb_performance(bounds):
    entity = "ic-ai-safety"
    project = "Cartpole"
    id_tag = "sa_rew"
    wandb_login()
    api = wandb.Api()
    entity, project = entity, project
    runs = api.runs(entity + "/" + project,
                    filters={"$and": [{"tags": id_tag, "state": "finished"}]}
                    )

    summary_list, config_list, name_list, state_list = [], [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

        # state_list.append(run._state)

    # runs_df = pd.DataFrame(
    #     {"summary": summary_list, "config": config_list, "name": name_list}
    # )
    #
    # runs_df.to_csv("project.csv")

    all_dicts = []
    for s, c, n in zip(summary_list, config_list, name_list):
        s_dict = {f"summary.{k}": v for k, v in s.items()}
        s_dict.update({f"config.{k}": v for k, v in c.items()})
        s_dict["name"] = n
        # s_dict["state"] = st
        all_dicts.append(s_dict)

    df = pd.DataFrame.from_dict(all_dicts)

    # hp = [x for x in df.columns if re.search("config", x)]
    # hp = [h for h in hp if h not in ["config.wandb_tags"]]
    # hp = [h for h in hp if len(df[h].unique()) > 1]

    hp = [f"config.{b}" for b in bounds.keys()]
    dfn = df[hp].select_dtypes(include='number')
    X = dfn
    y = df["summary.mean_episode_rewards"]
    return X, y

    # crs = dfn.corrwith(df["summary.mean_episode_rewards"])
    # crs = crs[pd.notna(crs)]
    #
    # from sklearn import tree
    #
    # clf = tree.DecisionTreeRegressor()
    # clf = clf.fit(X, y)
    # clf.predict([[1, 1]])
    #
    #
    #
    # crs[crs.abs().argmax()]
    # from numpy.polynomial import Polynomial as Poly
    # from numpy.polynomial import polynomial as P
    # x = dfn["config.gamma"]
    # y = df["summary.mean_episode_rewards"]
    # c = P.polyfit(x, y, deg=2)
    # c[0]
    # c[1]
    # c[2]
    #
    # # x = -c[1]/(2*c[2])
    # # c[2]*x**2 + c[1]*x + c[0]
    #
    #
    # Poly.fit(x, df["summary.mean_episode_rewards"], deg=2)
    #
    # np.polyfit(x, df["summary.mean_episode_rewards"], deg=2)
    #
    #
    #
    # plt.scatter(x, y, color='crimson', label='given points')
    #
    # # poly = np.polyfit(x, y, deg=2, rcond=None, full=False, w=None, cov=False)
    # poly = P.polyfit(x, y, deg=2)
    # xs = np.linspace(min(x), max(x), 100)
    # ys = poly[2] * xs ** 2 + poly[1] * xs + poly[0]
    # plt.plot(xs, ys, color='dodgerblue', label=f'$({poly[2]:.2f})x^2+({poly[1]:.2f})x + ({poly[0]:.2f})$')
    # plt.legend()
    # plt.show()

    print("x")

    # logdirs = np.unique([cfg["logdir"] for cfg in config_list])
    # logdir = 'logs/train/acrobot/test/2024-04-25__10-03-20__seed_6033'
    # flt = np.array([cfg["logdir"] == logdir for cfg in config_list])
    #
    # [bool(re.search("acrobot", cfg["logdir"])) for cfg in config_list]
    # [summary.get("problem_name", "") == "acrobot" for summary in summary_list]
    #
    # machines = list(filter(lambda summary: summary.get("problem_name", "") == "acrobot", summary_list))


def select_next_hyperparameters(X, y, bounds):
    [b.sort() for b in bounds.values()]
    bound_array = np.array([b for b in bounds.values()])

    X.loc[[f"config.{b}" for b in bounds.keys()]]
    next_params = bayesian_optimisation(X, y, bounds)


def run_next_hyperparameters(hparams):
    pass


def main(bounds):
    X, y = get_wandb_performance(bounds.keys())

    hparams = select_next_hyperparameters(X, y, bounds)

    run_next_hyperparameters(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    bounds = {
        "gamma": [0.99999, 0.0],
        "lmbda": [0.0, 0.99999],
        "val_epochs": [1, 10],
        "dyn_epochs": [1, 10],
        "dr_epochs": [1, 10],
        "learning_rate": [1e-8, 1e-1],
        "t_learning_rate": [1e-8, 1e-1],
        "dr_learning_rate": [1e-8, 1e-1],
        "n_envs": [16, 128],
        "n_steps": [128, 512],
        "n_rollouts": [1, 5],
        "temperature": [1e-8, 1e-1],
        "rew_coef": [0.0, 10.],
        "done_coef": [0.0, 10.],
        "output_dim": [1, 256],
        "depth": [1, 12],
        "mid_weight": [1, 2048],
        "n_minibatch": [16, 64],
    }
    while True:
        main(bounds)
