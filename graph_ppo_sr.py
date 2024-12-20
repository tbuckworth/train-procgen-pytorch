import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.naive_bayes import CategoricalNB
from torch import nn
from torch.distributions import Categorical

import wandb
from common.model import NBatchPySRTorch

from double_graph_sr import find_model, trial_agent_mean_reward
from graph_sr import fine_tune, get_pysr_dir, load_pysr_to_torch
from helper_local import add_symbreg_args, wandb_login, n_params, get_project, create_symb_dir_if_exists
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


def generate_data_supervised(agent, env, n):
    def predict(Obs):
        with torch.no_grad():
            Obs = torch.FloatTensor(Obs).to(agent.policy.device)
            Logits, a_out, m_out = agent.policy.forward_fine_tune(Obs)
            act = Logits.sample().detach().cpu().numpy()
        return Logits.logits, act, Obs, a_out, m_out

    Obs = env.reset()
    Logits, act, Obs, A_out, M_out = predict(Obs)
    act[::2] = np.random.randint(0, env.action_space.n, len(act))[::2]
    while len(Logits) < n:
        obs, rew, done, info = env.step(act)
        logits, act, obs, a_out, m_out = predict(obs)
        act[::2] = np.random.randint(0, env.action_space.n, len(act))[::2]

        Logits = torch.cat([Logits, logits], axis=0)
        Obs = torch.cat([Obs, obs], axis=0)
        A_out = torch.cat([A_out, a_out], axis=0)
        M_out = torch.cat([M_out, m_out], axis=0)

    return Obs, Logits, A_out, M_out


def fine_tune_supervised(ns_agent, nn_agent, env, test_env, args, ftdir, a_coef=1., m_coef=1000.):
    mean_rewards = trial_agent_mean_reward(ns_agent, env, "", n=args.n_tests, seed=args.seed, print_results=False, reset=False)
    val_mean_rewards = trial_agent_mean_reward(ns_agent, test_env, "", n=args.n_tests,
                                               seed=args.seed, print_results=False, reset=False)
    nc = args.num_checkpoints
    save_every = args.num_timesteps//nc
    checkpoints = [(i+1)*save_every for i in range(nc)] + [args.num_timesteps - 2]
    checkpoints.sort()
    t = 0
    i = 0

    with torch.no_grad():
        x, y, a_out, m_out = generate_data_supervised(nn_agent, env, args.batch_size)
        y_hat, a_out_hat, m_out_hat = ns_agent.policy.graph.forward_fine_tune(x)
        l_loss = nn.MSELoss()(y, y_hat)
        a_loss = nn.MSELoss()(a_out, a_out_hat)
        m_loss = nn.MSELoss()(m_out, m_out_hat)
        loss = l_loss + a_loss * a_coef + m_loss * m_coef

    wandb.log({
        'timesteps': t,
        'loss': loss.item(),
        'l_loss': l_loss.item(),
        'a_loss': a_loss.item(),
        'm_loss': m_loss.item(),
        'mean_reward': mean_rewards,
        'val_mean_reward': val_mean_rewards
    })
    optimizer = torch.optim.Adam(ns_agent.policy.parameters(), lr=args.learning_rate)
    for _ in range(args.num_timesteps // args.batch_size):
        x, y, a_out, m_out = generate_data_supervised(nn_agent, env, args.batch_size)
        # losses, a_losses, m_losses = [], [], []
        for _ in range(args.epoch):
            y_hat, a_out_hat, m_out_hat = ns_agent.policy.graph.forward_fine_tune(x)
            l_loss = nn.MSELoss()(y, y_hat)
            a_loss = nn.MSELoss()(a_out, a_out_hat)
            m_loss = nn.MSELoss()(m_out, m_out_hat)

            loss = l_loss + a_loss * a_coef + m_loss * m_coef
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # losses += [loss.item()]
            # a_losses += [a_loss.item()]
            # m_losses += [m_loss.item()]


        t += len(x)
        mean_rewards = trial_agent_mean_reward(ns_agent, env, "", n=args.n_tests,
                                               seed=args.seed, print_results=False, reset=False)
        val_mean_rewards = trial_agent_mean_reward(ns_agent, test_env, "", n=args.n_tests,
                                                   seed=args.seed, print_results=False, reset=False)

        log = {
            'timesteps': t,
            'loss': loss.item(),
            'l_loss': l_loss.item(),
            'a_loss': a_loss.item(),
            'm_loss': m_loss.item(),
            'mean_reward': mean_rewards,
            'val_mean_reward': val_mean_rewards
        }
        print(log)
        wandb.log(log)
        if t > checkpoints[i]:
            print("Saving model.")
            torch.save({'model_state_dict': ns_agent.policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       ftdir + '/model_' + str(t) + '.pth')
            i += 1
            p = Categorical(logits=y).probs.detach().cpu().numpy()
            p_hat = Categorical(logits=y_hat).probs.detach().cpu().numpy()
            plt.scatter(p[:, 0], p_hat[:, 0])
            plt.show()



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
    if args.load_pysr:
        symbdir = args.symbdir
        save_file = "symb_reg.csv"
    else:
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
    if args.load_pysr:
        msgdir = get_pysr_dir(symbdir, "msg")
        actdir = get_pysr_dir(symbdir, "act")
        msg_torch = load_pysr_to_torch(msgdir)
        act_torch = load_pysr_to_torch(actdir)
    else:
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

    # supervised learning:
    _, env, _, test_env = load_nn_policy(logdir, n_envs=100)
    ftdir = os.path.join(symbdir, "fine_tune")
    if not os.path.exists(ftdir):
        os.mkdir(ftdir)

    fine_tune_supervised(ns_agent, nn_agent, env, test_env, args, ftdir)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()
    args.logdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033"

    args.iterations = 1

    args.load_pysr = False
    # args.symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-08-27__10-39-50"
    args.symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-08-27__19-55-01"

    args.model_selection = "accuracy"
    args.maxsize = 50
    args.binary_operators = ["+", "-", "*", "greater", "/"]
    args.unary_operators = ["sin", "relu", "log", "exp", "sign", "sqrt", "square"]
    args.device = "gpu" if torch.cuda.is_available() else "cpu"
    args.learning_rate = 1e-3
    args.ncycles_per_iteration = 4000
    args.n_tests = 100
    args.batch_size = 1000
    args.num_checkpoints = 10
    args.num_timesteps = int(1e7)
    args.epoch = 100
    run_graph_ppo_sr(args)
