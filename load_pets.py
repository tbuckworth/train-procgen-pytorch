from pysr import PySRRegressor
import argparse
import os
import re

import mbrl.util.common as common_util
import numpy as np
import torch
from mbrl import models

from common.env.env_constructor import get_pets_env_constructor
from common.model import NBatchPySRTorch, NBatchPySRTorchMult
from double_graph_sr import find_model
from graph_sr import get_pysr_dir
from helper_local import get_latest_file_matching, get_config, DictToArgs, add_symbreg_args, create_symb_dir_if_exists
from pets.pets import generate_pets_cfg_dict, load_pets_agent
from symbreg.agents import PetsSymbolicAgent
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns

from symbreg.extra_mappings import get_extra_torch_mappings


def load_pets_dynamics_model(logdir):
    model_dir = get_latest_file_matching(r"model_(\d*)", 1, logdir)
    args = DictToArgs(get_config(logdir))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env_cons = get_pets_env_constructor(args.env_name)
    env = env_cons(args, {})
    env_valid = env_cons(args, {}, is_valid=True)
    env.reset(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg = generate_pets_cfg_dict(args, device)
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    dynamics_model.load(model_dir)

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.cartpole
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.cartpole
    try:
        if args.use_custom_reward_fn:
            reward_fn = env.rew_func
            term_fn = env.done_func
    except Exception as e:
        pass

    model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)
    agent = load_pets_agent(args, device, model_env)

    # agent.set_trajectory_eval_fn()

    return agent, model_env, args, env, env_valid

def generate_data(agent, env, n):
    Obs, _ = env.reset()
    agent.reset()
    M_in, M_out, U_in, U_out = agent.sample(Obs)
    act = env.action_space.sample()
    ep_count = 0
    while ep_count < n:
        observation, rew, done, trunc, info = env.step(act)
        ep_count += done
        m_in, m_out, u_in, u_out = agent.sample(observation)
        act = env.action_space.sample()
        if ep_count % 3 == 0:
            act = agent.forward(observation)
        M_in = np.append(M_in, m_in, axis=1)
        M_out = np.append(M_out, m_out, axis=1)
        U_in = np.append(U_in, u_in, axis=1)
        U_out = np.append(U_out, u_out, axis=1)

    return M_in, M_out, U_in, U_out

def generate_data(agent, env, n):
    X, _ = env.reset()
    X = np.expand_dims(X,0)
    agent.reset()
    act = env.action_space.sample()
    A = np.expand_dims(act.copy(), 0)
    ep_count = 0
    while ep_count < n:
        x, rew, done, trunc, info = env.step(act)
        act = env.action_space.sample()
        if ep_count == 0 and not done:
            act = agent.forward(x)
        if done:
            ep_count += 1
            m_in, m_out, u_in, u_out, loss = agent.sample_pre_act(X, A)
            X = np.expand_dims(x, 0)
            A = np.expand_dims(act, 0)
            if ep_count == 1:
                M_in, M_out, U_in, U_out, Loss = m_in, m_out, u_in, u_out, loss
            else:
                M_in = np.append(M_in, m_in, axis=1)
                M_out = np.append(M_out, m_out, axis=1)
                U_in = np.append(U_in, u_in, axis=1)
                U_out = np.append(U_out, u_out, axis=1)
                Loss = np.append(Loss, loss, axis=1)

        X = np.append(X, np.expand_dims(x, 0), axis=0)
        A = np.append(A, np.expand_dims(act, 0), axis=0)

    return M_in, M_out, U_in, U_out, Loss

def trial_agent_mean_reward(agent, env, print_name, n=40, return_values=False, seed=0):
    episodes = 0
    obs, _ = env.reset(seed=seed)
    agent.reset()
    act = agent.forward(obs)
    cum_rew = 0
    episode_rewards = []
    while episodes < n:
        obs, rew, done, trunc, info = env.step(act)
        cum_rew += rew
        act = agent.forward(obs)
        if done:
            episodes += 1
            episode_rewards += [cum_rew]
            cum_rew = 0
    print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
    if return_values:
        return episode_rewards
    return np.mean(episode_rewards)

def pets_sr(sr_args):
    logdir = sr_args.logdir
    symbdir, save_file = create_symb_dir_if_exists(logdir)
    print(f"symbdir: '{symbdir}'")
    agent, model_env, args, env, env_valid = load_pets_dynamics_model(logdir)
    pets_agent = PetsSymbolicAgent(agent, model_env, args.num_particles)

    # generate env data
    obs, _ = env.reset(args.seed)
    m_in_f, m_out_f, u_in_f, u_out_f, loss = generate_data(pets_agent, env, n=5)

    # filter by loss:
    m_in, m_out, m_weight, u_in, u_out, u_weight = filter_data(m_in_f, m_out_f, u_in_f, u_out_f, loss)

    # do symbolic regression
    print("data generated")
    msgdir, _ = create_symb_dir_if_exists(symbdir, "msg")
    updir, _ = create_symb_dir_if_exists(symbdir, "upd")

    print("\nTransition Messenger:")
    msg_model, _ = find_model(m_in, m_out, msgdir, save_file, m_weight, sr_args)
    print("\nTransition Updater:")
    up_model, _ = find_model(u_in, u_out, updir, save_file, u_weight, sr_args)

    msg_torch = NBatchPySRTorch(msg_model.pytorch())
    up_torch = NBatchPySRTorchMult(up_model.pytorch())

    symb_agent = PetsSymbolicAgent(agent, model_env, args.num_particles, msg_torch, up_torch)

    neural_train_score = trial_agent_mean_reward(pets_agent, env, "Neural Train", n=10)
    symb_train_score = trial_agent_mean_reward(symb_agent, env, "Symbolic Train", n=10)
    neural_test_score = trial_agent_mean_reward(pets_agent, env_valid, "Neural Test", n=10)
    symb_test_score = trial_agent_mean_reward(symb_agent, env_valid, "Symbolic Test", n=10)

    print("next")

def load_pysr_model(msgdir):
    pickle_filename = os.path.join(msgdir, "symb_reg.pkl")
    return PySRRegressor.from_file(pickle_filename, extra_torch_mappings=get_extra_torch_mappings())




def load_pets_sr_and_test():
    symbdir = "logs/pets/cartpole_continuous/2024-08-05__02-43-29__seed_6033/symbreg/2024-08-14__10-10-48"
    logdir = re.search(r"(logs.*)symbreg", symbdir).group(1)

    agent, model_env, args, env, env_valid = load_pets_dynamics_model(logdir)

    pets_agent = PetsSymbolicAgent(agent, model_env, args.num_particles)

    msgdir = get_pysr_dir(symbdir, "msg")
    updir = get_pysr_dir(symbdir, "upd")

    msg_model = load_pysr_model(msgdir)
    up_model = load_pysr_model(updir)

    msg_torch = NBatchPySRTorch(msg_model.pytorch())
    up_torch = NBatchPySRTorchMult(up_model.pytorch())

    symb_agent = PetsSymbolicAgent(agent, model_env, args.num_particles, msg_torch, up_torch)

    obs, _ = env.reset(seed=0)

    # This is in case something has gone wrong, we can debug it
    try:
        symb_agent.forward(obs)
    except Exception as e:
        raise e

    neural_train_score = trial_agent_mean_reward(pets_agent, env, "Neural Train", n=10)
    symb_train_score = trial_agent_mean_reward(symb_agent, env, "Symbolic Train", n=10)
    neural_test_score = trial_agent_mean_reward(pets_agent, env_valid, "Neural Test", n=10)
    symb_test_score = trial_agent_mean_reward(symb_agent, env_valid, "Symbolic Test", n=10)

    print("next")




def filter_data(m_in, m_out, u_in, u_out, loss, max_obs=2000):
    elite_ensemble = loss.mean(-1).argmin()
    eloss = loss[elite_ensemble]
    rank = eloss.argsort()
    weight = eloss.max() - eloss
    w = np.expand_dims(weight, 1)
    m_in = m_in[elite_ensemble]
    m_out = m_out[elite_ensemble]
    u_in = u_in[elite_ensemble]
    u_out = u_out[elite_ensemble]

    m_in, m_out = keep_best(m_in, m_out, max_obs, rank)
    u_in, u_out = keep_best(u_in, u_out, max_obs, rank)

    m_weight = np.tile(w, (1, m_in.shape[1]))
    u_weight = np.tile(w, (1, u_in.shape[1]))
    flatten = lambda a: a.reshape(-1, a.shape[-1])
    m_in = flatten(m_in)
    m_out = flatten(m_out)
    m_weight = flatten(m_weight)
    u_in = flatten(u_in)
    u_out = flatten(u_out)
    u_weight = flatten(u_weight)
    return m_in, m_out, m_weight, u_in, u_out, u_weight


def keep_best(m_in, m_out, max_obs, rank):
    flt = rank <= max_obs / m_in.shape[1] - 1
    m_in = m_in[flt]
    m_out = m_out[flt]
    return m_in, m_out


if __name__ == "__main__":
    # load_pets_sr_and_test()
    # exit(0)
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    sr_args = parser.parse_args()
    sr_args.logdir = "logs/pets/cartpole_continuous/2024-08-05__02-43-29__seed_6033"
    # sr_args.logdir = "logs/pets/cartpole_continuous/2024-08-05__08-40-15__seed_6033"

    sr_args.iterations = 5

    sr_args.binary_operators = ["+", "-", "*", "greater", "/"]
    sr_args.unary_operators = ["sin", "relu", "log", "exp", "sign", "sqrt", "square"]

    sr_args.denoise = False
    sr_args.use_wandb = True
    sr_args.wandb_tags = ["test", "sr_pets0"]
    sr_args.wandb_name = "manual"
    # sr_args.fixed_nn = ["value"]
    # sr_args.populations = 24
    sr_args.model_selection = "best"
    sr_args.ncycles_per_iteration = 4000
    sr_args.seed = 0
    sr_args.bumper = True
    sr_args.loss_function = "mse"
    pets_sr(sr_args)
