import argparse
import re
import time
import traceback
from math import floor, log10

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

import wandb
from create_sh_files import add_training_args_dict, add_symbreg_args_dict

from gp import bayesian_optimisation
from helper_local import wandb_login, DictToArgs, get_project, add_symbreg_args, add_pets_args, add_espl_args, \
    add_training_args


def get_wandb_performance(hparams, project="Cartpole", id_tag="sa_rew", opt_metric="summary.mean_episode_rewards",
                          entity="ic-ai-safety"):
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
        y = df[opt_metric]
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

    hp = [f"config.{h}" for h in hparams if f"config.{h}" in df.columns]
    dfn = df[hp].select_dtypes(include='number')
    return dfn, y


def n_sig_fig(x, n):
    if x == 0 or isinstance(x, bool):
        return x
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))


def select_next_hyperparameters(X, y, bounds, greater_is_better=True):
    if bounds == {}:
        return {}
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
            param = bayesian_optimisation(x, yp, b, random_search=True, greater_is_better=greater_is_better)
            params += list(param)

        next_params = np.array(params)[np.argsort(idx)]

    int_params = [np.all([isinstance(x, int) for x in bounds[k]]) for k in col_order]
    bool_params = [np.all([isinstance(x, bool) for x in bounds[k]]) for k in col_order]
    next_params = [int(round(v, 0)) if i else v for i, v in zip(int_params, next_params)]
    next_params = [bool(v) if b else v for b, v in zip(bool_params, next_params)]

    hparams = {k: n_sig_fig(next_params[i], 3) for i, k in enumerate(col_order)}

    return hparams


def run_pets_hyperparameters(hparams):
    from pets.pets import run_pets
    parser = argparse.ArgumentParser()
    parser = add_pets_args(parser)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.update(hparams)
    run_pets(DictToArgs(args_dict))


def run_next_hyperparameters(hparams):
    from train import train_ppo
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    parser_dict = vars(parser.parse_args())
    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    return train_ppo(args)

    # from train import train_ppo
    # parser_dict = add_training_args_dict()
    # parser_dict.update(hparams)
    # args = DictToArgs(parser_dict)
    # train_ppo(args)


def run_hp_for_espl(hparams):
    from run_espl import run_espl_x_squared
    parser = argparse.ArgumentParser()
    parser = add_espl_args(parser)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.update(hparams)
    run_espl_x_squared(DictToArgs(args_dict))


def run_graph_hyperparameters(hparams):
    from graph_sr import run_graph_neurosymbolic_search
    parser_dict = add_training_args_dict()
    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    run_graph_neurosymbolic_search(args)


def run_double_graph_hyperparameters(hparams):
    from double_graph_sr import run_double_graph_neurosymbolic_search
    parser_dict = add_symbreg_args_dict()
    parser_dict.update(hparams)
    args = DictToArgs(parser_dict)
    run_double_graph_neurosymbolic_search(args)


def graph_ppo_multi_sr(hparams):
    from graph_ppo_sr_multi import run_graph_ppo_multi_sr
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.update(hparams)
    run_graph_ppo_multi_sr(DictToArgs(args_dict))


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


def optimize_hyperparams(bounds,
                         fixed,
                         project="Cartpole",
                         id_tag="sa_rew",
                         run_next=run_next_hyperparameters,
                         opt_metric="summary.mean_episode_rewards",
                         greater_is_better=True):
    strings = {k: v for k, v in fixed.items() if isinstance(v, list) and k != "wandb_tags"}
    string_select = {k: np.random.choice(v) for k, v in strings.items()}
    if "env_name" in string_select.keys():
        project = get_project(string_select["env_name"], fixed["exp_name"])
    try:
        X, y = get_wandb_performance(bounds.keys(), project, id_tag, opt_metric)
    except ValueError as e:
        print(f"Error from wandb:\n{e}\nPicking hparams randomly.")
        X, y = None, None

    if X is not None and np.prod(X.shape) == 0:
        X, y = None, None

    hparams = select_next_hyperparameters(X, y, bounds, greater_is_better)

    fh = fixed.copy()
    hparams.update(fh)
    hparams.update(string_select)

    hparams = {k: int(v) if isinstance(v, np.int64) else v for k, v in hparams.items()}
    try:
        run_next(hparams)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        wandb.finish(exit_code=-1)


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
        "output_dim": 43,  # [24, 64],
        "depth": 4,  # [2, 6],
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


def init_wandb(cfg, prefix="symbolic_graph"):
    name = np.random.randint(1e5)
    wandb_login()
    wb_resume = "allow"  # if args.model_file is None else "must"
    project = get_project(cfg["env_name"], cfg["exp_name"])
    wandb.init(project=project, config=cfg, sync_tensorboard=True,
               tags=cfg["wandb_tags"], resume=wb_resume, name=f"{prefix}-{name}")


def fine_tune_sr(hp_override):
    from graph_sr import fine_tune, load_sr_graph_agent

    symbdir = hp_override["symbdir"]
    logdir, ns_agent = load_sr_graph_agent(symbdir)

    # TODO: add newdir to this
    init_wandb(hp_override)

    del hp_override["symbdir"]
    fine_tune(ns_agent.policy, logdir, symbdir, hp_override)


def fine_tune_hparams():
    fixed = {
        "env_name": 'cartpole',
        "exp_name": 'symbreg',  # IMPORTANT!
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
        "env_name": 'cartpole_swing',
        "exp_name": 'symbreg',  # IMPORTANT!
        "param_name": 'double-graph',
        "device": "gpu",
        "seed": 6033,
        "maxsize": 40,
        "wandb_tags": ["ftdg01_swing", "double-graph", "graph-transition"],
        # "logdir": "logs/train/cartpole/2024-07-11__04-48-25__seed_6033",
        "logdir": "logs/train/cartpole_swing/test/2024-06-12__11-15-29__seed_6033",
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
        # "maxsize": [20, 50],
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


def pets_graph_transition_cartpole():
    fixed = {
        "env_name": 'cartpole_continuous',
        "exp_name": "",
        "seed": 6033,
        "wandb_tags": ["pgt3", "graph-transition", "residual"],
        "use_wandb": True,
        "deterministic": False,
        "residual": True,
        "trial_length": 500,
        "drop_same": True,
        "min_cart_mass": 1.0,
        "max_cart_mass": 1.0,
        "min_pole_mass": 0.1,
        "max_pole_mass": 0.1,
        "min_force_mag": 10.,
        "max_force_mag": 10.,
        "min_cart_mass_v": 1.0,
        "max_cart_mass_v": 1.0,
        "min_pole_mass_v": 0.1,
        "max_pole_mass_v": 0.1,
        "min_force_mag_v": 10.,
        "max_force_mag_v": 10.,
    }
    bounds = {
        # 'trial_length': [200, 200],
        'num_trials': [20, 20],
        'ensemble_size': [2, 5],
        'num_layers': [3, 7],
        'hid_size': [64, 512],
        # 'planning_horizon': [15, 15],
        # 'replan_freq': [1,1],
        # 'num_iterations': [5,5],
        # 'population_size': [500, 500],
        'num_particles': [5, 10],
        'learning_rate': [1e-6, 1e-2],
        # 'weight_decay': [5e-5, 5e-5],
        'num_epochs': [25, 75],
        # 'patience': [50, 50],
        # 'validation_ratio': [0.05, 0.05],
        # 'elite_ratio': [0.1, 0.1],
        # 'alpha': [0.1],
        'model_batch_size': [8, 32],
    }
    while True:
        project = get_project(fixed["env_name"], fixed["exp_name"])
        id_tag = fixed["wandb_tags"][0]
        optimize_hyperparams(bounds, fixed, project, id_tag, run_pets_hyperparameters, opt_metric="trial/total_reward")


def cartpole_graph_ppo():
    fixed = {
        "detect_nan": True,
        "env_name": 'cartpole_continuous',
        "exp_name": 'pure-graph',
        "param_name": 'graph-cartpole-cont',
        "device": "gpu",
        "num_timesteps": int(2e8),
        "seed": 6033,
        "use_gae": True,
        "clip_value": True,
        "wandb_tags": ["gc2"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": True,
        "anneal_temp": False,
        "entropy_coef": 0.01,
        # "n_envs": 64,
        # "n_steps": 4,
        # "n_minibatch": 1,
        # "output_dim": 24,#[24, 64],
        # "depth": 4,#[2, 6],
        "gamma": 0.985,
        "lmbda": 0.873,
        # "epoch": [1, 10],
        "learning_rate": 0.000527,
        "depth": 5,
        "mid_weight": 173,
    }
    bounds = {
        # "gamma": [0.9999, 0.8],
        # "lmbda": [0.0, 0.99999],
        # "epoch": [1, 10],
        # "learning_rate": [1e-8, 1e-3],
        # # "n_envs": [64],
        # # "n_steps": [256],
        # "depth": [2, 6],
        # "mid_weight": [8, 256],
    }
    while True:
        project = get_project(fixed["env_name"], fixed["exp_name"])
        id_tag = fixed["wandb_tags"][0]
        optimize_hyperparams(bounds, fixed, project, id_tag, run_next_hyperparameters)


def humanoid_graph_ppo():
    fixed = {
        "detect_nan": True,
        "env_name": 'humanoid',  # 'cartpole_continuous',
        "exp_name": 'pure-graph',
        "param_name": 'graph-ant',
        "device": "gpu",
        "num_timesteps": int(1e8),
        "seed": 6033,
        "use_gae": True,
        "clip_value": True,
        "wandb_tags": ["gh0"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": True,
        "anneal_temp": False,
        # "entropy_coef": 0.,
        # "n_envs": 6,
        # "learning_rate": 1e-4,
        # "n_steps": 4096,
        # "n_minibatch": 128,
        # "output_dim": 24,#[24, 64],
        # "depth": 4,#[2, 6],
        "gamma": 0.99,
        "lmbda": 0.95,
        "epoch": 10,
        "simple_scaling": True,
    }
    bounds = {
        # "simple_scaling": [False, True],
        "entropy_coef": [0.001, 0.02],
        # "gamma": [0.9999, 0.8],
        # "lmbda": [0.0, 0.99999],
        # "epoch": [1, 10],
        "learning_rate": [1e-4, 1e-3],
        # "n_envs": [64],
        # "n_steps": [256],
        "depth": [2, 6],
        "mid_weight": [16, 256],
    }
    # try somehting like these for humanoid:
    # nsteps: 4096
    # nminibatches: 64
    # lam: 0.95
    # gamma: 0.99
    # noptepochs: 10
    # ent_coef: 0.0
    # lr: lambda f: 1e-4 * f
    # cliprange: 0.1
    # value_network: copy
    # num_hidden: 512
    # num_layers: 4
    run_forever(bounds, fixed, run_next_hyperparameters)


def graph_ppo_sr_ft():
    fixed = {
        "env_name": 'cartpole',
        "exp_name": 'symbreg',  # IMPORTANT!
        "param_name": 'graph',
        "device": "gpu",
        "seed": 6033,
        "wandb_tags": ["gpp2"],
        # "logdir": "logs/train/cartpole/2024-07-11__04-48-25__seed_6033",
        "logdir": "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033",
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
        'load_pysr': False,
        'sequential': True,
        'stochastic': False,
        'min_mse': True,
        'num_checkpoints': 10,
        'n_tests': 40,
        'load_ft': False,
    }
    bounds = {
        "data_size": [1000, 5000],
        "iterations": [1, 100],
        "populations": [15, 40],
        "procs": [4, 16],
        "ncycles_per_iteration": [4000, 6000],
        "num_timesteps": [int(1e4), int(1e5)],
        "epoch": [10, 1000],
        "learning_rate": [1e-2, 1e-2],
        'batch_size': [100, 1100],
        "maxsize": [20, 60],
    }
    project = get_project(fixed["env_name"], fixed["exp_name"])
    id_tag = fixed["wandb_tags"][0]
    while True:
        optimize_hyperparams(bounds, fixed, project, id_tag, graph_ppo_multi_sr)


def graph_ppo_sr_ft_continuous():
    fixed = {
        "env_name": 'cartpole',
        "exp_name": 'symbreg',  # IMPORTANT!
        "param_name": 'graph',
        "device": "gpu",
        "seed": 6033,
        "wandb_tags": ["gpp-cont1"],
        "logdir": "logs/train/cartpole_continuous/pure-graph/2024-09-08__00-59-06__seed_6033",
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
        'load_pysr': False,
        'sequential': True,
        'min_mse': True,
        'num_checkpoints': 2,
        'n_tests': 40,
        'stochastic': False,
        'ncycles_per_iteration': 4000,
    }
    bounds = {
        "data_size": [1000, 5000],
        "iterations": [1, 60],
        "populations": [15, 40],
        "procs": [4, 16],
        # "ncycles_per_iteration": [4000, 6000],
        "num_timesteps": [int(1e4), int(1e5)],
        "epoch": [10, 1000],
        "learning_rate": [5e-3, 15e-3],
        'batch_size': [100, 1100],
        "maxsize": [20, 60],
    }
    run_forever(bounds, fixed, graph_ppo_multi_sr)


def run_forever(bounds, fixed, run_func):
    project = get_project(fixed["env_name"], fixed["exp_name"])
    id_tag = fixed["wandb_tags"][0]
    fixed["original_start"] = time.asctime()
    while True:
        optimize_hyperparams(bounds, fixed, project, id_tag, run_func, greater_is_better=True)


def espl_x_squared():
    fixed = {
        "wandb_tags": ["espl_hp0"],
    }
    bounds = {
        "epochs": [10000, 100000],
        "sample_num": [1, 20],
        "lr": [5e-3, 15e-3],
        'hard_gum': [True, False],
        "hard_ratio": [0.5, .9],
        'other_loss_scale': [1, 10000],
        'target_ratio': [0.00001, 0.01],
        'spls': [0.1, 1.0],
        'target_temp': [0.01, 0.00001],
    }
    project = 'espl'
    id_tag = fixed["wandb_tags"][0]
    fixed["original_start"] = time.asctime()
    while True:
        optimize_hyperparams(bounds, fixed, project, id_tag, run_hp_for_espl, opt_metric='summary.ood_loss')


def espo_cartpole():
    fixed = {
        "detect_nan": True,
        "env_name": 'cartpole_continuous',
        "exp_name": 'espo-graph',
        "param_name": 'eql-graph',
        "device": "gpu",
        "num_timesteps": int(1e8),
        "seed": 6033,
        "use_gae": True,
        "clip_value": True,
        "wandb_tags": ["eql0"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": True,
        "anneal_temp": False,
        # "entropy_coef": 0.,
        # "n_envs": 6,
        # "learning_rate": 1e-4,
        # "n_steps": 4096,
        # "n_minibatch": 128,
        # "output_dim": 24,#[24, 64],
        # "depth": 4,#[2, 6],
        # "gamma": 0.99,
        # "lmbda": 0.95,
        # "epoch": 10,
        "simple_scaling": True,
    }
    bounds = {
        # # "simple_scaling": [False, True],
        # "entropy_coef": [0.001, 0.02],
        # # "gamma": [0.9999, 0.8],
        # # "lmbda": [0.0, 0.99999],
        # # "epoch": [1, 10],
        # "learning_rate": [1e-4, 1e-3],
        # # "n_envs": [64],
        # # "n_steps": [256],
        # "depth": [2, 6],
        # "mid_weight": [16, 256],
    }
    run_forever(bounds, fixed, run_next_hyperparameters)


def ipl_cartpole():
    fixed = {
        "detect_nan": True,
        "env_name": 'cartpole',  # ['cartpole',"cartpole_swing","mountain_car","acrobot", "cartpole_continuous"],
        "exp_name": 'IPL',
        "param_name": 'ipl_cartpole',
        "device": "gpu",
        "num_timesteps": int(4e6),
        "seed": [6033, 42],
        "wandb_tags": ["entropy_modifier"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": False,
        "anneal_temp": False,
        "use_greedy_env": True,
        "learned_gamma": True,
        "learned_temp": False,  # [True, False],
        "reward_incentive": False,  # [True, False],
        "adv_incentive": False,  # [True, False],
        # "alpha_learning_rate": 2e-4,
        # "n_envs": 6,
        # "learning_rate": 1e-4,
        # "n_steps": 4096,
        # "n_minibatch": 128,
        # "output_dim": 24,#[24, 64],
        # "depth": 4,#[2, 6],
        # "gamma": 0.99,
        # "lmbda": 0.95,
        # good cartpole one: (with 4 epochs)
        "learning_rate": 0.000374,
        "epoch": 4,
        "entropy_modified": True,
        # "entropy_coef": 1,
        # "alpha": [0, 0.01, 0.1, 0.25, 0.5],
    }
    bounds = {
        # # "gamma": [0.9999, 0.8],
        # # "lmbda": [0.0, 0.99999],
        # "epoch": [4],
        # "learning_rate": [1e-11, 1e-6],
        # "alpha_learning_rate":[1e-5, 1e-3],
        # "target_entropy_coef": [0.01, 0.5],
        # # "n_envs": [64],
        # # "n_steps": [256],
        # "depth": [2, 6],
        # "mid_weight": [16, 256],
    }
    run_forever(bounds, fixed, run_next_hyperparameters)


def ipl_icm_cartpole():
    fixed = {
        "detect_nan": True,
        "env_name": ['cartpole', "cartpole_swing", "mountain_car", "acrobot", "cartpole_continuous"],
        "exp_name": 'IPL_ICM',
        "param_name": 'ipl_icm_cartpole',
        "device": "gpu",
        "num_timesteps": int(3e6),
        "seed": [6033, 0],
        "wandb_tags": ["icm4"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": False,
        "anneal_temp": False,
        "use_greedy_env": True,
        "learned_gamma": True,
        "learned_temp": False,
        "reward_incentive": False,
        "adv_incentive": False,
        # "alpha_learning_rate": 2e-4,
        # "n_envs": 6,
        # "learning_rate": 1e-4,
        # "n_steps": 4096,
        # "n_minibatch": 128,
        # "output_dim": 24,#[24, 64],
        # "depth": 4,#[2, 6],
        # "gamma": 0.99,
        # "lmbda": 0.95,
        # good cartpole one: (with 4 epochs)
        "learning_rate": 0.000374,
        "epoch": 4,
        # "alpha": [1, 0, 0.01, 0.1, 0.25, 0.5],
        "n_imagined_actions": 0,
        "zv_loss_coef": 0,
        "novelty_loss_coef": 0,
        # "beta": 0.1,
        "beta": 0,
        "separate_icm": True,
    }
    bounds = {
        # "zv_loss_coef": [0., 1.],
        # "novelty_loss_coef": [0., 1.],
        # "beta": [0.1],
        # "alpha": [0.01, 0.5],
        # # "gamma": [0.9999, 0.8],
        # # "lmbda": [0.0, 0.99999],
        # "epoch": [1, 4],
        # "learning_rate": [1e-11, 1e-6],
        # "alpha_learning_rate":[1e-5, 1e-3],
        # "target_entropy_coef": [0.01, 0.5],
        # # "n_envs": [64],
        # # "n_steps": [256],
        # "depth": [2, 6],
        # "mid_weight": [16, 256],
    }
    run_forever(bounds, fixed, run_next_hyperparameters)


def ipl_coinrun():
    fixed = {
        "detect_nan": True,
        "env_name": 'coinrun',
        "exp_name": 'Coinrun',
        "param_name": 'hard-500-ipl',
        "device": "gpu",
        "num_timesteps": int(2e7),
        "seed": 6033,
        "wandb_tags": ["ipl0"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": False,
        "anneal_temp": False,
        "use_greedy_env": True,
        "learned_gamma": True,
        "start_level": 431,
        "num_levels": 1,
        "accumulate_all_grads": False,
        # "n_envs": 6,
        # "learning_rate": 1e-4,
        # "n_steps": 4096,
        "n_minibatch": 16,
        "entropy_modified": True,
        # "output_dim": 24,#[24, 64],
        # "depth": 4,#[2, 6],
        # "gamma": 0.99,
        # "lmbda": 0.95,
        # "epoch": 10,
    }
    bounds = {
        # "alpha": [0.1, 0.01, 0.2],
        # # "gamma": [0.9999, 0.8],
        # # "lmbda": [0.0, 0.99999],
        "epoch": [4],
        "learning_rate": [0.000374],  # 1e-10],#1e-8, 5e-4],
        # # "n_envs": [64],
        # # "n_steps": [256],
        # "depth": [2, 6],
        # "mid_weight": [16, 256],
    }
    run_forever(bounds, fixed, run_next_hyperparameters)


def ppo():
    fixed = {
        "detect_nan": False,
        "env_name": 'cartpole',
        "exp_name": 'Cartpole',
        "param_name": 'mlpmodel',
        "device": "gpu",
        "num_timesteps": int(1e7),
        "seed": [6033, 0, 42, 50, 81],
        "wandb_tags": ["deep narrow"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": True,
        "anneal_temp": False,
        # "wandb_group": "logsum_exp_independent",
        "logsumexp_logits_is_v": [False],
        "use_greedy_env": False,
        "entropy_coef": [0, 0.02],
    }
    bounds = {
        # "alpha": [0.1, 0.01, 0.2],
        # # "gamma": [0.9999, 0.8],
        # # "lmbda": [0.0, 0.99999],
        # "epoch": [4],
        # "learning_rate": [0.000374],#1e-10],#1e-8, 5e-4],
        # # "n_envs": [64],
        # # "n_steps": [256],
        "depth": [10, 30],
        "mid_weight": [2, 8],
    }
    run_forever(bounds, fixed, run_next_hyperparameters)


def goal_seeker():
    fixed = {
        "detect_nan": False,
        "env_name": 'cartpole',
        "exp_name": 'Cartpole',
        "param_name": 'goal-seeker-mlp',
        "device": "gpu",
        "num_timesteps": int(1e7),
        "seed": [6033, 0, 42, 50, 81],
        "wandb_tags": ["gs1"],
        "use_wandb": True,
        "mirror_env": False,
        "use_valid_env": False,
        "use_greedy_env": False,

        "use_planning_to_act": [True, False],
    }
    bounds = {
        "entropy_coef": [0, 0.02],
        "goal_loss_coef": [0.8, 1.2],
        "distance_loss_coef": [0.1, 1.],
        "forward_loss_coef": [0.8, 1.2],
        "action_loss_coef": [0.8, 1.2],
        # "alpha": [0.1, 0.01, 0.2],
        # # "gamma": [0.9999, 0.8],
        # # "lmbda": [0.0, 0.99999],
        # "epoch": [4],
        # "learning_rate": [0.000374],#1e-10],#1e-8, 5e-4],
        # # "n_envs": [64],
        # # "n_steps": [256],
        # "depth": [2, 6],
        # "mid_weight": [16, 256],
    }
    run_forever(bounds, fixed, run_next_hyperparameters)


if __name__ == "__main__":
    # import faulthandler
    # faulthandler.enable()
    # ppo()
    # goal_seeker()
    ppo()
    # ipl_coinrun()
    # ipl_cartpole()
    # espl_x_squared()
    # cartpole_graph_ppo()
    # graph_ppo_sr_ft_continuous()
    # humanoid_graph_ppo()
    # double_graph_symbreg_ft_hparams()
    # pets_graph_transition_cartpole()
