import os
import re

import numpy as np
from pysr import PySRRegressor

from common.env.env_constructor import get_env_constructor
from helper_local import get_config, DictToArgs, get_actions_from_all, map_actions_to_values
from symbolic_regression import load_nn_policy, test_agent_mean_reward
from symbreg.agents import NeuralAgent

def make_obs_mountain_car(obs):
    return np.expand_dims(np.concatenate((obs, [0.0025])), 0)

def gymnasium_agent_test(agent, env, print_name, n=40, return_values=False):
    print(print_name)
    episodes = 0
    obs, info = env.reset()
    act = agent.forward(make_obs_mountain_car(obs))
    cum_rew = 0
    episode_rewards = []
    while episodes < n:
        obs, rew, done, trunc, info = env.step(int(act[0]))
        cum_rew += rew
        act = agent.forward(make_obs_mountain_car(obs))
        if done:
            episodes += 1
            episode_rewards += cum_rew
            cum_rew = 0
    print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
    if return_values:
        return episode_rewards
        # return {"mean":np.mean(episode_rewards), "std":np.std(episode_rewards)}
    return np.mean(episode_rewards)

if __name__ == "__main__":
    # symbdir = "logs/train/acrobot/test/2024-05-01__12-22-24__seed_6033/symbreg/2024-05-02__02-06-38"

    # symbdir = "logs/train/cartpole_swing/test/2024-05-01__14-19-53__seed_6033/symbreg/2024-05-03__04-30-33"
    # !!!!!!!!!!!!!!
    symbdir = "logs/train/cartpole/test/2024-05-01__11-17-16__seed_6033/symbreg/2024-05-03__11-14-03"
    symbdir = "logs/train/mountain_car/test/2024-05-03__15-46-58__seed_6033/symbreg/2024-05-13__10-50-45"
    symbdir = "logs/train/mountain_car/test/2024-05-03__15-46-58__seed_6033/symbreg/2024-05-14__00-45-59"

    n_envs = 2
    n_rounds = 10
    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    logdir = re.search(r"(logs.*)symbreg", symbdir).group(1)
    pysr_model = PySRRegressor.from_file(pickle_filename)
    orig_cfg = get_config(logdir)
    env_name = orig_cfg["env_name"]
    cfg = get_config(symbdir)
    env_cons = get_env_constructor(orig_cfg["env_name"])
    args = DictToArgs(cfg)
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, args.n_envs)

    actions = get_actions_from_all(env)
    action_mapping = map_actions_to_values(actions)

    ns_agent = symbolic_agent_constructor(pysr_model, policy, args.stochastic, action_mapping)
    nn_agent = NeuralAgent(policy)

    train_params = env.get_params().copy()
    test_params = test_env.get_params().copy()
    # pole_params = train_params.copy()
    # pole_params["min_pole_length"] = test_params["min_pole_length"]
    # pole_params["max_pole_length"] = test_params["max_pole_length"]

    train_params["n_envs"] = n_envs
    test_params["n_envs"] = n_envs
    # pole_params["n_envs"] = n_envs
    train_env = env_cons(DictToArgs({"render": True, "seed": 0}), train_params, False)
    test_env = env_cons(DictToArgs({"render": True, "seed": 0}), test_params, False)
    # pole_env = env_cons(DictToArgs({"render": True, "seed": 0}), pole_params, False)

    # hyperparameters = {
    #     "degrees": 12,
    #     "h_range": 2.4,
    #     "min_gravity": 10.07256063,
    #     "max_gravity": 10.07256063,
    #     "min_pole_length": 1.65905418,
    #     "max_pole_length": 1.65905418,
    #     "min_cart_mass": 1.38712133,
    #     "max_cart_mass": 1.38712133,
    #     "min_pole_mass": 0.12735904,
    #     "max_pole_mass": 0.12735904,
    #     "min_force_mag": 10.,
    #     "max_force_mag": 10.,
    # }
    #
    hyperparameters = {
        "min_gravity": 0.0025,
        "max_gravity": 0.0025
    }
    cust_env = env_cons(DictToArgs({"render": True, "seed": 0}), hyperparameters, False)

    # test_agent_mean_reward(nn_agent, cust_env, "NeuralAgent Cust", 5)
    test_env.reset(seed=42)
    test_agent_mean_reward(ns_agent, cust_env, "SymbolicAgent Cust", 20)

    # import gymnasium as gym
    # orig_env = gym.make("MountainCar-v0", render_mode="human")
    # gymnasium_agent_test(ns_agent, orig_env, "SymbolicAgent Orig", 20)



    # test_agent_mean_reward(nn_agent, train_env, "NeuralAgent Train", 5)
    # train_env.reset(seed=0)
    test_agent_mean_reward(ns_agent, train_env, "SymbolicAgent Train", 5)
    #
    # test_agent_mean_reward(nn_agent, test_env, "NeuralAgent Test", 5)
    # test_env.reset(seed=42)
    test_agent_mean_reward(ns_agent, test_env, "SymbolicAgent Test", 5)
