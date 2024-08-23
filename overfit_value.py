import argparse

import numpy as np
import torch
import wandb
from torch.nn import MSELoss

from common.env.env_constructor import get_env_constructor
from common.model import GraphTransitionModel, NBatchPySRTorch, GraphValueModel
from common.storage import BasicStorage
from double_graph_sr import find_model, create_symb_dir_if_exists
from helper_local import add_symbreg_args, DictToArgs, n_params
from hyperparameter_optimization import init_wandb
from symbreg.agents import flatten_batches_to_numpy
from train import create_logdir_train


def collect_transition_samples_value(env, args, storage, model, device):
    obs = env.reset()
    for _ in range(args.data_size // env.n_envs):
        obs_t = torch.FloatTensor(obs).to(device=device)
        with torch.no_grad():
            value = model(obs_t)
        act = np.array([env.action_space.sample() for _ in range(env.n_envs)])
        new_obs, rew, done, info = env.step(act)
        storage.store(obs, act, rew, done, info, value.cpu().numpy())
        obs = new_obs
    obs_t = torch.FloatTensor(obs).to(device=device)
    with torch.no_grad():
        value = model(obs_t)
    storage.store_last(obs, value.cpu().numpy())
    storage.compute_estimates(args.gamma, args.lmbda, True, True)



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
        policy="random",
        epochs=1000,
        resample_every=1000,
        env_name="cartpole",
        exp_name="overfit",
        seed=0,
        n_envs=2,
        data_size=1000,
        mini_batch_size=128,
        sr_every=100,
        learning_rate=1e-6,
        s_learning_rate=1e-4,
        depth=4,
        mid_weight=256,
        latent_size=1,
        weights=None,
        wandb_tags=[],
        gamma=.999,
        lmbda=0.95,
        eps_clip=0.2,
        clip_value=False,
    )
    a = DictToArgs(cfg)
    env_cons = get_env_constructor(a.env_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = env_cons(None, {"n_envs": a.n_envs})
    env_v = env_cons(None, {"n_envs": a.n_envs}, is_valid=True)
    observation_shape = env.observation_space.shape
    in_channels = observation_shape[0]

    storage = BasicStorage(observation_shape, a.data_size // a.n_envs, a.n_envs, device)
    storage.reset()
    storage_v = BasicStorage(observation_shape, a.data_size // a.n_envs, a.n_envs, device)
    storage_v.reset()

    if a.policy == "random":
        # TODO:
        pass
    elif a.policy == "optimal":
        # TODO:
        pass
    else:
        raise NotImplementedError(f"policy must be one of 'random','optimal'. Not {a.policy}")

    model = GraphValueModel(in_channels, a.depth, a.mid_weight, a.latent_size, device)
    model_v = GraphValueModel(in_channels, a.depth, a.mid_weight, a.latent_size, device)
    symb_model = GraphValueModel(in_channels, a.depth, a.mid_weight, a.latent_size, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    optimizer_v = torch.optim.Adam(model_v.parameters(), lr=a.learning_rate)

    model.to(device)
    model_v.to(device)

    sr_params = {
        "binary_operators": ["+", "-", "greater", "*", "/"],
        "unary_operators": ["sin", "relu", "log", "exp", "sign", "sqrt", "square"],
        "iterations": 5,
        "maxsize": 50,
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
    init_wandb(cfg, prefix="GVN")

    for epoch in range(a.epochs):
        # collect transition samples
        collect_transition_samples_value(env, a, storage, model, device)
        collect_transition_samples_value(env_v, a, storage_v, model_v, device)

        val_model_loss_v = optimize_value(a, storage_v, model_v, optimizer_v, clip_value=a.clip_value)
        val_model_loss = optimize_value(a, storage, model_v, None, False, clip_value=a.clip_value)

        loss = optimize_value(a, storage, model, optimizer, clip_value=a.clip_value)
        loss_v = optimize_value(a, storage_v, model, None, False, clip_value=a.clip_value)

        # do sr

        if epoch == 0 or np.isnan(s_loss) or (loss < s_loss and epoch % a.sr_every == 0):
            m_in, m_out, u_in, u_out = collect_messages(model, storage, a)
            weights = None#get_weights(a, nobs, nobs_guess)

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

        s_loss = optimize_value(a, storage, symb_model, s_optimizer, clip_value=a.clip_value)
        s_loss_v = optimize_value(a, storage_v, symb_model, None, False, clip_value=a.clip_value)

        if use_wandb:
            wandb.log({"Epoch": epoch,
                       "GNN Loss": loss,
                       "GNN Validation Loss": loss_v,
                       "Symb Loss": s_loss,
                       "Symb Validation Loss": s_loss_v,
                       "Valid-GNN Loss": val_model_loss,
                       "Valid-GNN Validation Loss": val_model_loss_v,
                       })
    wandb.finish()




def optimize_value(args, storage, model, optimizer, update=True, clip_value=True):
    generator = storage.fetch_train_generator(args.mini_batch_size, False)

    loss_list = []
    for sample in generator:
        obs_batch, nobs_batch, act_batch, done_batch, \
            old_value_batch, return_batch, adv_batch, rew_batch = sample

        value_batch = model(obs_batch)

        # Clipped Bellman-Error
        clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-args.eps_clip,
                                                                                      args.eps_clip)
        v_surr1 = (value_batch - return_batch).pow(2)
        v_surr2 = (clipped_value_batch - return_batch).pow(2)
        loss = 0.5 * torch.max(v_surr1, v_surr2).mean()
        if not clip_value:
            loss = v_surr1.mean()

        if update:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_list += [loss.item()]
    mean_loss = np.mean(loss_list)
    return mean_loss




def get_weights(a, nobs, nobs_guess):
    if a.weights is None:
        weights = None
    elif a.weights == "neg_exp":
        weights = flatten_batches_to_numpy(torch.exp(-((nobs_guess - nobs) ** 2)))
    else:
        raise NotImplementedError(f"weights must be None or 'neg_exp', not {a.weights}")
    return weights


def collect_messages(model, storage, args):
    generator = storage.fetch_train_generator(args.mini_batch_size, False)
    mi = mo = ui = uo = None
    with torch.no_grad():
        for sample in generator:
            obs_batch, nobs_batch, act_batch, done_batch, \
                old_value_batch, return_batch, adv_batch, rew_batch = sample
            n, x = model.prep_input(obs_batch)
            msg_in = model.vectorize_for_message_pass(n, x)
            messages = model.messenger(msg_in)
            h, u = model.vec_for_update(messages, x)

            m_in = flatten_batches_to_numpy(msg_in)
            m_out = flatten_batches_to_numpy(messages)
            u_in = flatten_batches_to_numpy(h)
            u_out = flatten_batches_to_numpy(u)
            if mi is None:
                mi = m_in
                mo = m_out
                ui = u_in
                uo = u_out
            else:
                mi = np.append(mi, m_in, axis=0)
                mo = np.append(mo, m_out, axis=0)
                ui = np.append(ui, u_in, axis=0)
                uo = np.append(uo, u_out, axis=0)
    return mi, mo, ui, uo


if __name__ == "__main__":
    overfit()
