import numpy as np
import torch
from torch.nn import MSELoss

from common.env.env_constructor import get_env_constructor
from common.model import GraphTransitionModel
from double_graph_sr import find_model
from symbreg.agents import PureGraphSymbolicAgent, DummyPolicy, flatten_batches_to_numpy


def collect_transition_samples(env, n, device):
    obs = env.reset()
    x = np.expand_dims(obs, axis=1)
    a = None
    d = None
    while np.prod(x.shape[:2])<n:
        act = np.array([env.action_space.sample() for _ in range(env.n_envs)])
        obs, rew, done, info = env.step(act)
        a = append_or_create(a, act)
        d = append_or_create(d, done)
        x = append_or_create(x, obs)

    obs = torch.FloatTensor(x[:,:-1]).to(device=device)
    nobs = torch.FloatTensor(x[:,1:]).to(device=device)
    acts = torch.FloatTensor(a[:,:-1]).to(device=device)
    dones = torch.BoolTensor(d[:,:-1]).to(device=device)
    return obs[~dones], nobs[~dones], acts[~dones]



def append_or_create(a, act):
    a_tmp = np.expand_dims(act, axis=1)
    if a is None:
        a = a_tmp
    a = np.append(a, a_tmp, axis=1)
    return a


def overfit():
    epochs = 1000
    env_cons = get_env_constructor("cartpole")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_envs = 2
    env = env_cons(None, {"n_envs": n_envs})
    env_v = env_cons(None, {"n_envs": n_envs}, is_valid=True)
    observation_shape = env.observation_space.shape
    in_channels = observation_shape[0]
    model = GraphTransitionModel(in_channels, depth=4, mid_weight=256, latent_size=1, device=device)

    model.to(device)

    # collect transition samples
    obs, nobs, acts = collect_transition_samples(env, n=100, device=device)
    obs_v, nobs_v, acts_v = collect_transition_samples(env_v, n=100, device=device)

    for epoch in range(epochs):
        with torch.no_grad():
            nobs_guess_v = model(obs_v, acts_v)
            loss_v = MSELoss()(nobs_guess_v, nobs_v)

        nobs_guess = model(obs, acts)
        loss = MSELoss()(nobs_guess, nobs)

        #do sr
        if epoch % 100 == 0:
            m_in, m_out, u_in, u_out = collect_messages(acts, model, obs)
            weights = flatten_batches_to_numpy(torch.exp(-((nobs_guess-nobs)**2)))
            weights = None
            print("\nTransition Messenger:")
            msg_model, _ = find_model(m_in, m_out, msgdir, save_file, weights, args)
            print("\nTransition Updater:")
            up_model, _ = find_model(u_in, u_out, updir, save_file, weights, args)


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
