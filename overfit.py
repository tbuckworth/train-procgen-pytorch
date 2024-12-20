import argparse

import numpy as np
import torch
import wandb
from torch.nn import MSELoss

from common.env.env_constructor import get_env_constructor
from common.model import GraphTransitionModel, NBatchPySRTorch, MLPTransitionModel
from double_graph_sr import find_model
from helper_local import add_symbreg_args, DictToArgs, n_params, create_symb_dir_if_exists
from hyperparameter_optimization import init_wandb
from symbreg.agents import flatten_batches_to_numpy
from train import create_logdir_train


def collect_transition_samples_value(env, n, storage, model):
    obs = env.reset()
    for _ in range(n // env.n_envs):
        with torch.no_grad():
            value = model(obs)
        act = np.array([env.action_space.sample() for _ in range(env.n_envs)])
        new_obs, rew, done, info = env.step(act)
        storage.store(obs, act, rew, done, info, value)
        obs = new_obs


def collect_transition_samples(env, n, device):
    obs = env.reset()
    x = np.expand_dims(obs, axis=1)
    a = None
    d = None
    while np.prod(x.shape[:2]) < n:
        act = np.array([env.action_space.sample() for _ in range(env.n_envs)])
        obs, rew, done, info = env.step(act)
        a = append_or_create(a, act)
        d = append_or_create(d, done)
        x = append_or_create(x, obs)

    obs = torch.FloatTensor(x[:, :-1]).to(device=device)
    nobs = torch.FloatTensor(x[:, 1:]).to(device=device)
    acts = torch.FloatTensor(a[:, :-1]).to(device=device)
    dones = torch.BoolTensor(d[:, :-1]).to(device=device)
    return obs[~dones], nobs[~dones], acts[~dones]


def append_or_create(a, act):
    a_tmp = np.expand_dims(act, axis=1)
    if a is None:
        a = a_tmp
    a = np.append(a, a_tmp, axis=1)
    return a


def overfit(use_wandb=True):
    cfg = dict(
        epochs=1000,
        resample_every=1000,
        env_name="cartpole",
        exp_name="overfit",
        seed=0,
        n_envs=2,
        data_size=1000,
        sr_every=100,
        learning_rate=1e-5,
        mlp_learning_rate=1e-4,
        s_learning_rate=1e-2,
        depth=4,
        mid_weight=256,
        latent_size=1,
        weights=None,
        wandb_tags=[],
        gamma=.998,
        lmbda=0.735,
    )
    a = DictToArgs(cfg)
    env_cons = get_env_constructor(a.env_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = env_cons(None, {"n_envs": a.n_envs})
    env_v = env_cons(None, {"n_envs": a.n_envs}, is_valid=True)
    observation_shape = env.observation_space.shape
    in_channels = observation_shape[0]

    model = GraphTransitionModel(in_channels, a.depth, a.mid_weight, a.latent_size, device)
    symb_model = GraphTransitionModel(in_channels, a.depth, a.mid_weight, a.latent_size, device)
    mlp = MLPTransitionModel(in_channels,a.depth,a.mid_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=a.mlp_learning_rate)

    model.to(device)
    mlp.to(device)

    sr_params = {
        "binary_operators": ["+", "-", "greater", "*", "/"],
        "unary_operators": ["sin", "relu", "log", "exp", "sign", "sqrt", "square"],
        "iterations": 1,
    }

    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)
    args = parser.parse_args()
    parser_dict = vars(args)
    parser_dict.update(sr_params)
    sr_args = DictToArgs(parser_dict)

    logdir = create_logdir_train("", a.env_name, a.exp_name, a.seed)
    symbdir, save_file = create_symb_dir_if_exists(logdir)

    cfg.update(parser_dict)
    init_wandb(cfg, prefix="GNN")


    for epoch in range(a.epochs):
        # collect transition samples
        if epoch % a.resample_every == 0:
            obs, nobs, acts = collect_transition_samples(env, n=a.data_size, device=device)
            if epoch == 0:
                obs_v, nobs_v, acts_v = collect_transition_samples(env_v, n=a.data_size, device=device)

        loss, loss_v, nobs_guess = calc_losses(acts, acts_v, model, nobs, nobs_v, obs, obs_v)

        mlp_loss, mlp_loss_v, _ = calc_losses(acts, acts_v, mlp, nobs, nobs_v, obs, obs_v)

        # do sr

        if epoch == 0 or s_loss.isnan() or (loss.item() < s_loss.item() and epoch % a.sr_every == 0):
            m_in, m_out, u_in, u_out = collect_messages(acts, model, obs)
            weights = get_weights(a, nobs, nobs_guess)

            msgdir, _ = create_symb_dir_if_exists(symbdir, "msg")
            updir, _ = create_symb_dir_if_exists(symbdir, "upd")

            idx = np.random.permutation(len(m_in))[:sr_args.data_size]
            print("\nTransition Messenger:")
            msg_model, _ = find_model(m_in[idx], m_out[idx], msgdir, save_file, weights, sr_args)

            idx = np.random.permutation(len(u_in))[:sr_args.data_size]
            print("\nTransition Updater:")
            up_model, _ = find_model(u_in[idx], u_out[idx], updir, save_file, weights, sr_args)

            msg_torch = NBatchPySRTorch(msg_model.pytorch())
            up_torch = NBatchPySRTorch(up_model.pytorch())

            symb_model.messenger = msg_torch
            symb_model.updater = up_torch
            print(f"Neural Parameters: {n_params(model)}")
            print(f"Symbol Parameters: {n_params(symb_model)}")
            s_optimizer = torch.optim.Adam(symb_model.parameters(), lr=a.s_learning_rate)

        s_loss, s_loss_v, _ = calc_losses(acts, acts_v, symb_model, nobs, nobs_v, obs, obs_v)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mlp_loss.backward()
        mlp_opt.step()
        mlp_opt.zero_grad()

        s_loss.backward()
        s_optimizer.step()
        s_optimizer.zero_grad()

        if use_wandb:
            wandb.log({"Epoch": epoch,
                       "GNN Loss": loss.item(),
                       "GNN Validation Loss": loss_v.item(),
                       "Symb Loss": s_loss.item(),
                       "Symb Validation Loss": s_loss_v.item(),
                       "MLP Loss": mlp_loss.item(),
                       "MLP Validation Loss": mlp_loss_v.item(),
                       })
    wandb.finish()


def calc_losses(acts, acts_v, model, nobs, nobs_v, obs, obs_v):
    with torch.no_grad():
        nobs_guess_v = model(obs_v, acts_v)
        loss_v = MSELoss()(nobs_guess_v, nobs_v)
    nobs_guess = model(obs, acts)
    loss = MSELoss()(nobs_guess, nobs)
    return loss, loss_v, nobs_guess


def get_weights(a, nobs, nobs_guess):
    if a.weights is None:
        weights = None
    elif a.weights == "neg_exp":
        weights = flatten_batches_to_numpy(torch.exp(-((nobs_guess - nobs) ** 2)))
    else:
        raise NotImplementedError(f"weights must be None or 'neg_exp', not {a.weights}")
    return weights


def collect_messages(acts, model, obs):
    with torch.no_grad():
        n, x = model.prep_input(obs)
        msg_in = model.vectorize_for_message_pass(acts, n, x)
        messages = model.messenger(msg_in)
        h, u = model.vec_for_update(messages, x)

        m_in = flatten_batches_to_numpy(msg_in)
        m_out = flatten_batches_to_numpy(messages)
        u_in = flatten_batches_to_numpy(h)
        u_out = flatten_batches_to_numpy(u)
    return m_in, m_out, u_in, u_out


if __name__ == "__main__":
    overfit()
