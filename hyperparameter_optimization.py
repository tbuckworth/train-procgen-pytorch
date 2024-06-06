import argparse
import re
from math import floor, log10

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import wandb
from create_sh_files import add_training_args_dict
from gp import bayesian_optimisation
from helper_local import wandb_login, DictToArgs
from train import train_ppo


def get_wandb_performance(hparams, project="Cartpole", id_tag="sa_rew", entity="ic-ai-safety"):
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

    all_dicts = []
    for s, c, n in zip(summary_list, config_list, name_list):
        s_dict = {f"summary.{k}": v for k, v in s.items()}
        s_dict.update({f"config.{k}": v for k, v in c.items()})
        s_dict["name"] = n
        all_dicts.append(s_dict)

    df = pd.DataFrame.from_dict(all_dicts)

    # hp = [x for x in df.columns if re.search("config", x)]
    # hp = [h for h in hp if h not in ["config.wandb_tags"]]
    # hp = [h for h in hp if len(df[h].unique()) > 1]

    hp = [f"config.{h}" for h in hparams]
    dfn = df[hp].select_dtypes(include='number')
    X = dfn
    y = df["summary.mean_episode_rewards"]
    return X, y


def n_sig_fig(x, n):
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def select_next_hyperparameters(X, y, bounds):
    [b.sort() for b in bounds.values()]
    col_order = [re.sub(r"config\.", "", k) for k in X.columns]
    bound_array = np.array([bounds[k] for k in col_order])

    next_params = bayesian_optimisation(X.to_numpy(), y.to_numpy(), bound_array)
    int_params = [np.all([isinstance(x, int) for x in bounds[k]]) for k in col_order]
    next_params = [int(round(v, 0)) if i else v for i, v in zip(int_params, next_params)]

    hparams = {k: n_sig_fig(next_params[i], 3) for i, k in enumerate(col_order)}

    return hparams


def run_next_hyperparameters(hparams):
    parser_dict = add_training_args_dict()
    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    train_ppo(args)


def main(bounds, fixed, project="Cartpole", id_tag="sa_rew"):
    X, y = get_wandb_performance(bounds.keys(),  project, id_tag)

    hparams = select_next_hyperparameters(X, y, bounds)

    fh = fixed.copy()
    hparams.update(fh)

    run_next_hyperparameters(hparams)


if __name__ == "__main__":
    fixed = {
        "env_name": 'cartpole',
        "param_name": 'graph-transition',
        "device": "gpu",
        "num_timesteps": int(2e6),
        "seed": 6033,
        "use_gae": True,
        "clip_value": False,
        "wandb_tags": ["graph-transition", "sa_rew", "gam_lam"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": True,
        "anneal_temp": False,
    }
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
        main(bounds, fixed, "Cartpole", "sa_rew")
