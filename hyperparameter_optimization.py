import re
from math import floor, log10

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

import wandb
from create_sh_files import add_training_args_dict
from double_graph_sr import run_double_graph_neurosymbolic_search
from gp import bayesian_optimisation
from graph_sr import fine_tune, load_sr_graph_agent, run_graph_neurosymbolic_search
from helper_local import wandb_login, DictToArgs, get_project
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
    try:
        y = df["summary.mean_episode_rewards"]
    except KeyError:
        return None, None

    flt = pd.notna(y)
    df = df[flt]
    y = y[flt]
    if len(df) == 0:
        return None, None
    # hp = [x for x in df.columns if re.search("config", x)]
    # hp = [h for h in hp if h not in ["config.wandb_tags"]]
    # hp = [h for h in hp if len(df[h].unique()) > 1]

    hp = [f"config.{h}" for h in hparams]
    dfn = df[hp].select_dtypes(include='number')
    return dfn, y


def n_sig_fig(x, n):
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def select_next_hyperparameters(X, y, bounds):
    [b.sort() for b in bounds.values()]
    if X is None:
        bound_array = np.array([[x[0], x[-1]] for x in bounds.values()])
        next_params = np.random.uniform(bound_array[:, 0], bound_array[:, 1], (bound_array.shape[0]))
        col_order = bounds.keys()
    else:
        col_order = [re.sub(r"config\.", "", k) for k in X.columns]
        bo = [bounds[k] for k in col_order]

        bound_array = np.array([[x[0], x[-1]] for x in bo])

        xp = X.to_numpy()
        yp = y.to_numpy()

        params = []
        idx = np.random.permutation(len(X.columns))

        n_splits = np.ceil(len(idx) / 2)
        xs = np.array_split(xp[:, idx], n_splits, axis=1)
        bs = np.array_split(bound_array[idx], n_splits, axis=0)

        for x, b in zip(xs, bs):
            param = bayesian_optimisation(x, yp, b, random_search=True)
            params += list(param)

        next_params = np.array(params)[np.argsort(idx)]

    int_params = [np.all([isinstance(x, int) for x in bounds[k]]) for k in col_order]
    next_params = [int(round(v, 0)) if i else v for i, v in zip(int_params, next_params)]

    hparams = {k: n_sig_fig(next_params[i], 3) for i, k in enumerate(col_order)}

    return hparams


def run_next_hyperparameters(hparams):
    parser_dict = add_training_args_dict()
    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    train_ppo(args)

def run_graph_hyperparameters(hparams):
    parser_dict = add_training_args_dict()
    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    run_graph_neurosymbolic_search(args)

def run_double_graph_hyperparameters(hparams):
    parser_dict = add_training_args_dict()
    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    run_double_graph_neurosymbolic_search(args)

def inspect_hparams(X, y, bounds, fixed):
    cuttoff = 495
    cuttoff = y.max()
    cuttoff = 485
    X["flt"] = y >= cuttoff
    h_stats = X.pivot_table(columns="flt", aggfunc=["mean", "std", "count"])
    t, p = ttest_ind_from_stats(h_stats[('mean', True)], h_stats[('std', True)], h_stats['count', True],
                                h_stats[('mean', False)], h_stats[('std', False)], h_stats['count', False])
    h_stats[('p_val', None)] = p
    h_stats = h_stats.drop(columns=[("count", False), ("count", True)])
    print(h_stats)

    pass


def optimize_hyperparams(bounds, fixed, project="Cartpole", id_tag="sa_rew", run_next=run_next_hyperparameters):
    X, y = get_wandb_performance(bounds.keys(), project, id_tag)

    hparams = select_next_hyperparameters(X, y, bounds)

    fh = fixed.copy()
    hparams.update(fh)
    # run_next(hparams)
    try:
        run_next(hparams)
    except Exception as e:
        print(e)

def cartpole_graph_hyperparams():
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
        # "lmbda": .95,
        # "val_epochs": 8,#[1, 10],
        # "dyn_epochs": 3,#[1, 10],
        # "dr_epochs": 3,#[1, 10],
        # "learning_rate": 1e-4,#[1e-8, 1e-3],
        # "t_learning_rate": 5e-4,#[1e-8, 1e-3],
        # "dr_learning_rate": 5e-5,#[1e-8, 1e-3],
        "n_envs": 64,
        "n_steps": 256,
        "n_rollouts": 3,
        # "temperature": 1e-5,#[1e-8, 1e-2],
        # "rew_coef": 1.,#[0.1, 10.],
        # "done_coef": 1.,#[0.1, 10.],
        # "output_dim": 24,#[24, 64],
        # "depth": 4,#[2, 6],
    }
    bounds = {
        "gamma": [0.9999, 0.8],
        "lmbda": [0.0, 0.99999],
        "val_epochs": [1, 10],
        "dyn_epochs": [1, 10],
        "dr_epochs": [1, 10],
        "learning_rate": [1e-8, 1e-3],
        "t_learning_rate": [1e-8, 1e-3],
        "dr_learning_rate": [1e-8, 1e-3],
        # "n_envs": [64],
        # "n_steps": [256],
        # "n_rollouts": [3],
        "temperature": [1e-8, 1e-2],
        "rew_coef": [0.1, 10.],
        "done_coef": [0.1, 10.],
        "output_dim": [24, 64],
        "depth": [2, 6],
    }
    while True:
        optimize_hyperparams(bounds, fixed, "Cartpole", "sa_rew", run_next_hyperparameters)


def cartpole_full_graph_hyperparams():
    fixed = {
        "env_name": 'cartpole',
        "exp_name": "",
        "param_name": 'full-graph-transition',
        "device": "gpu",
        "num_timesteps": int(4e6),
        "seed": 6033,
        "use_gae": True,
        "clip_value": False,
        "wandb_tags": ["fg01", "full-graph-transition", "graph-transition"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": True,
        "anneal_temp": False,
        # "lmbda": .998,
        # "gamma": 0.735,
        "learning_rate": 0.000532,
        # "temperature": 0.00545,

        "n_envs": 64,
        "n_steps": 256,
        "n_rollouts": 3,
        # "temperature": 1e-5,#[1e-8, 1e-2],
        # "rew_coef": 1.,#[0.1, 10.],
        # "done_coef": 1.,#[0.1, 10.],
        "output_dim": 43,#[24, 64],
        "depth": 4,#[2, 6],
    }
    bounds = {
        "gamma": [0.9999, 0.8],
        "lmbda": [0.0, 0.99999],
        "epoch": [2, 10],
        # "learning_rate": [1e-8, 1e-3],
        # "n_envs": [64],
        # "n_steps": [256],
        # "n_rollouts": [3],
        "temperature": [1e-4, 1e-2],
        "rew_coef": [0.1, 10.],
        "done_coef": [0.1, 10.],
        "t_coef": [0.1, 10.],
        "value_coef": [0.1, 10.],
        # "output_dim": [24, 64],
        # "depth": [2, 6],
    }
    while True:
        project = get_project(fixed["env_name"], fixed["exp_name"])
        id_tag = fixed["wandb_tags"][0]
        optimize_hyperparams(bounds, fixed, project, id_tag, run_next_hyperparameters)



def init_wandb(cfg):
    name = np.random.randint(1e5)
    wandb_login()
    wb_resume = "allow"  # if args.model_file is None else "must"
    project = get_project(cfg["env_name"], cfg["exp_name"])
    wandb.init(project=project, config=cfg, sync_tensorboard=True,
                   tags=cfg["wandb_tags"], resume=wb_resume, name=f"symbolic_graph-{name}")


def fine_tune_sr(hp_override):
    symbdir = hp_override["symbdir"]
    logdir, ns_agent = load_sr_graph_agent(symbdir)

    #TODO: add newdir to this
    init_wandb(hp_override)

    del hp_override["symbdir"]
    fine_tune(ns_agent.policy, logdir, symbdir, hp_override)


def fine_tune_hparams():
    fixed = {
        "env_name": 'cartpole',
        "exp_name": 'symbreg', # IMPORTANT!
        "param_name": 'graph-transition',
        "device": "gpu",
        "num_timesteps": int(2e6),
        "seed": 6033,
        "wandb_tags": ["ft033", "graph-transition"],
        # "val_epochs": 0,
    }
    bounds = {
        "num_timesteps": [int(1e5), int(2e6)],
        # "gamma": [0.9999, 0.8],
        # "lmbda": [0.0, 0.99999],
        "val_epochs": [1, 10],
        # "dyn_epochs": [1, 10],
        "dr_epochs": [1, 10],
        "learning_rate": [1e-8, 1e-3],
        # "t_learning_rate": [1e-8, 1e-3],
        "dr_learning_rate": [1e-8, 1e-3],
        # "n_envs": [64],
        # "n_steps": [256],
        # "n_rollouts": [3],
        # "temperature": [1e-8, 1e-2],
        "rew_coef": [0.1, 10.],
        "done_coef": [0.1, 10.],
    }
    project = get_project(fixed["env_name"], fixed["exp_name"])
    id_tag = fixed["wandb_tags"][0]
    while True:
        fixed["symbdir"] = "logs/train/cartpole/test/2024-06-08__00-54-02__seed_6033/symbreg/2024-06-11__11-29-55"
        # fixed["symbdir"] = "logs/train/cartpole/test/2024-06-11__10-31-41__seed_6033/symbreg/2024-06-13__14-06-20"
        optimize_hyperparams(bounds, fixed, project, id_tag, fine_tune_sr)

def graph_symbreg_ft_hparams():
    fixed = {
        "env_name": 'cartpole',
        "exp_name": 'symbreg',  # IMPORTANT!
        "param_name": 'graph-transition',
        "device": "gpu",
        "seed": 6033,

        "wandb_tags": ["ft034", "graph-transition"],
        "logdir": "logs/train/cartpole/test/2024-06-08__00-54-02__seed_6033",
        "timeout_in_seconds": 3600 * 10,
        "n_envs": 2,
        "denoise": False,
        "binary_operators": ["+", "-", "greater", "*", "/"],
        "unary_operators": ["sin", "relu", "log", "exp", "sign", "sqrt", "square"],
        "use_wandb": True,
        "bumper": False,
        "model_selection": "accuracy",
        "loss_function": 'mse',
        "fixed_nn": [],
        "weight_metric": None,

    }
    bounds = {
        "data_size": [1000, 20000],
        "iterations": [1, 70],
        "populations": [15, 40],
        "procs": [4, 16],
        "ncycles_per_iteration": [4000, 6000],
        "num_timesteps": [int(1e5), int(2e6)],

        # "gamma": [0.9999, 0.8],
        # "lmbda": [0.0, 0.99999],
        "val_epochs": [1, 10],
        "dyn_epochs": [1, 10],
        "dr_epochs": [1, 10],
        "learning_rate": [1e-8, 1e-3],
        "t_learning_rate": [1e-8, 1e-3],
        "dr_learning_rate": [1e-8, 1e-3],
        # "n_envs": [64],
        # "n_steps": [256],
        # "n_rollouts": [3],
        # "temperature": [1e-8, 1e-2],
        "rew_coef": [0.1, 10.],
        "done_coef": [0.1, 10.],
    }
    project = get_project(fixed["env_name"], fixed["exp_name"])
    id_tag = fixed["wandb_tags"][0]
    while True:
        optimize_hyperparams(bounds, fixed, project, id_tag, run_graph_hyperparameters)


def double_graph_symbreg_ft_hparams():
    fixed = {
        "env_name": 'cartpole',
        "exp_name": 'symbreg',  # IMPORTANT!
        "param_name": 'double-graph',
        "device": "gpu",
        "seed": 6033,

        "wandb_tags": ["ftdg01", "double-graph", "graph-transition"],
        "logdir": "logs/train/cartpole/2024-07-11__04-48-25__seed_6033",
        "timeout_in_seconds": 3600 * 10,
        "n_envs": 2,
        "denoise": False,
        "binary_operators": ["+", "-", "greater", "*", "/"],
        "unary_operators": ["sin", "relu", "log", "exp", "sign", "sqrt", "square"],
        "use_wandb": True,
        "bumper": False,
        "model_selection": "accuracy",
        "loss_function": 'mse',
        "weight_metric": None,
    }
    bounds = {
        "data_size": [1000, 20000],
        "iterations": [1, 70],
        "populations": [15, 40],
        "procs": [4, 16],
        "ncycles_per_iteration": [4000, 6000],
        "num_timesteps": [int(1e5), int(2e6)],

        # "gamma": [0.9999, 0.8],
        # "lmbda": [0.0, 0.99999],
        "val_epochs": [1, 10],
        "dyn_epochs": [1, 10],
        "learning_rate": [1e-8, 1e-3],
        "t_learning_rate": [1e-8, 1e-3],
        # "n_envs": [64],
        # "n_steps": [256],
        # "n_rollouts": [3],
        # "temperature": [1e-8, 1e-2],
    }
    project = get_project(fixed["env_name"], fixed["exp_name"])
    id_tag = fixed["wandb_tags"][0]
    while True:
        optimize_hyperparams(bounds, fixed, project, id_tag, run_double_graph_hyperparameters)


def cartpole_double_graph_hyperparams():
    fixed = {
        "env_name": 'cartpole',
        "exp_name": "",
        "param_name": 'double-graph',
        "device": "gpu",
        "num_timesteps": int(3e6),
        "seed": 6033,
        "use_gae": True,
        "clip_value": False,
        "wandb_tags": ["dg01", "double-graph", "graph-transition"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": True,
        "anneal_temp": False,
        "gamma": .998,
        "lmbda": 0.735,
        "n_minibatch": 32,
        # "learning_rate": 0.000532,
        "t_learning_rate": 0.000511,
        "temperature": 0.00545,
        "dyn_epochs": 9,
        "n_envs": 64,
        "n_steps": 256,
        "n_rollouts": 3,
        "output_dim": 43,
        "depth": 4,
    }
    bounds = {
        # "gamma": [0.9999, 0.8],
        # "lmbda": [0.0, 0.99999],
        "val_epochs": [2, 10],
        "learning_rate": [9e-4, 1e-2],
        # "n_envs": [64],
        # "n_steps": [256],
        # "n_rollouts": [3],
        # "temperature": [1e-4, 1e-2],
        # "value_coef": [0.1, 10.],
        # "output_dim": [24, 64],
        # "depth": [2, 6],
    }
    while True:
        project = get_project(fixed["env_name"], fixed["exp_name"])
        id_tag = fixed["wandb_tags"][0]
        optimize_hyperparams(bounds, fixed, project, id_tag, run_next_hyperparameters)

if __name__ == "__main__":
    double_graph_symbreg_ft_hparams()


