import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch

import wandb
from common.model import NBatchPySRTorch

from double_graph_sr import create_symb_dir_if_exists, find_model
from graph_sr import fine_tune
from helper_local import add_symbreg_args, wandb_login, n_params, get_project
from symbolic_regression import load_nn_policy


def generate_data(agent, env, n):
    Obs = env.reset()
    M_in, M_out, A_in, A_out = agent.sample(Obs)
    act = agent.forward(Obs)
    act[::2] = np.random.randint(0, env.action_space.n, len(act))[::2]
    while len(M_in) < n:
        observation, rew, done, info = env.step(act)
        m_in, m_out, a_in, a_out = agent.sample(observation)
        act = agent.forward(observation)
        act[::2] = np.random.randint(0, env.action_space.n, len(act))[::2]

        M_in = np.append(M_in, m_in, axis=0)
        M_out = np.append(M_out, m_out, axis=0)
        A_in = np.append(A_in, a_in, axis=0)
        A_out = np.append(A_out, a_out, axis=0)

    return M_in, M_out, A_in, A_out


def run_graph_ppo_sr(args):
    logdir = args.logdir
    n_envs = args.n_envs
    data_size = args.data_size
    hp_override = {
        "device": args.device,
        "seed": args.seed,
        # "epoch": args.epoch,
        "learning_rate": args.learning_rate,
    }
    symbdir, save_file = create_symb_dir_if_exists(logdir)
    cfg = vars(args)
    np.save(os.path.join(symbdir, "config.npy"), cfg)

    wandb_name = args.wandb_name
    if args.wandb_name is None:
        wandb_name = f"graph-ppo-sr{np.random.randint(1e5)}"

    if args.use_wandb:
        wandb_login()
        name = wandb_name
        wb_resume = "allow"  # if args.model_file is None else "must"
        project = get_project("cartpole", "symbreg")
        cfg["symbdir"] = symbdir
        if args.wandb_group is not None:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name, group=args.wandb_group)
        else:
            wandb.init(project=project, config=cfg, sync_tensorboard=True,
                       tags=args.wandb_tags, resume=wb_resume, name=name)

    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs)
    nn_agent = symbolic_agent_constructor(policy)
    m_in, m_out, a_in, a_out = generate_data(nn_agent, env, int(data_size))

    print("data generated")
    weights = None
    msgdir, _ = create_symb_dir_if_exists(symbdir, "msg")
    actdir, _ = create_symb_dir_if_exists(symbdir, "act")

    print("\nMessenger:")
    msg_model, _ = find_model(m_in, m_out, msgdir, save_file, weights, args)
    print("\nActor:")
    act_model, _ = find_model(a_in, a_out, actdir, save_file, weights, args)

    msg_torch = NBatchPySRTorch(msg_model.pytorch())
    act_torch = NBatchPySRTorch(act_model.pytorch())

    try:
        wandb.log({
            "messenger": msg_model.get_best().equation,
            "actor": act_model.get_best().equation,
        })
    except Exception as e:
        pass


    ns_agent = symbolic_agent_constructor(copy.deepcopy(policy), msg_torch, act_torch)

    print(f"Neural Parameters: {n_params(nn_agent.policy)}")
    print(f"Symbol Parameters: {n_params(ns_agent.policy)}")


    fine_tuned_policy = fine_tune(ns_agent.policy, logdir, symbdir, hp_override)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()
    args.logdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033"

    args.iterations = 1

    args.binary_operators = ["+", "-", "*", "greater", "/"]
    args.unary_operators = ["sin", "relu", "log", "exp", "sign", "sqrt", "square"]
    args.device = "gpu" if torch.cuda.is_available() else "cpu"
    args.learning_rate = 1e-2
    run_graph_ppo_sr(args)
