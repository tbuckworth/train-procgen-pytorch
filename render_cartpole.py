import numpy as np

from cartpole.cartpole_pre_vec import CartPoleVecEnv
from common.env.env_constructor import get_env_constructor
from discrete_env.acrobot_pre_vec import AcrobotVecEnv
from discrete_env.mountain_car_pre_vec import MountainCarVecEnv
from helper_local import sigmoid, sample_from_sigmoid, DictToArgs, map_actions_to_values, get_actions_from_all, match, \
    match_to_nearest
from symbolic_regression import load_nn_policy
from symbreg.agents import CustomModel, SymbolicAgent


def symbolic_regression_function(obs):
    x0, x1, x2, x3 = obs[0]
    if x0 > -1.62 * (3 * x2 + x3 + x1):
        return np.ones(len(obs))
    return np.zeros(len(obs))

def alt_func(obs):
    x0, x1, x2, x3 = obs.T
    p = sigmoid(x2/0.04 + x3)
    return sample_from_sigmoid(p)
    # return (0.57+5*x2+x3).round(decimals=0)

def acrobot_func(obs, action_mapping):
    #2523 lowest complexity
    x0, x1, x2, x3, x4, x5, x6 = obs.T
    out = np.sign((x1 / 0.28540868) + x5)
    return match(out, action_mapping)

def acrobot_best(obs, action_mapping):
    x0, x1, x2, x3, x4, x5, x6 = obs.T
    out = np.sign((x5 - (np.sin(np.sin(x3)) - x1)) - -0.011042326) * x6
    return match_to_nearest(out, action_mapping)


def greater(a, b):
    return (a > b).astype(np.int32)

def cartpole_swing_func(obs, action_mapping):
    x0, x1, x2, x3, x4, x5, x6, x7, x8 = obs.T
    out = np.sign(greater((x0 / x6) * -1.4575042, x5))
    return match_to_nearest(out, action_mapping)

def mountain_car_func(obs, action_mapping):
    x0, x1, x2 = obs.T
    return match_to_nearest(np.sign(x1), action_mapping)

# def temp()
#     episodes = 0
#     obs = env.reset()
#     act = agent.forward(obs)
#     episode_rewards = []
#     while episodes < n:
#         # This is for cartpole only!
#         ep_reward = env.env.n_steps.copy()
#         obs, rew, done, new_info = env.step(act)
#         act = agent.forward(obs)
#         if np.any(done):
#             episodes += np.sum(done)
#             episode_rewards += list(ep_reward[done])
#     print(f"{print_name}:\tEpisode:{episodes}\tMean Reward:{np.mean(episode_rewards):.2f}")
#     return np.mean(episode_rewards)

def mountain_car_goal_test():
    n = 100
    n_envs = 100
    for gp in [i/100 for i in range(0, 100)]:
        env = MountainCarVecEnv(n_envs=n_envs, goal_position=gp, sparse_rewards=True)
        action_mapping = map_actions_to_values(get_actions_from_all(env))
        episodes = 0
        obs = env.reset()
        act = mountain_car_func(obs, action_mapping)
        cum_rew = np.zeros(len(act))
        episode_rewards = []
        while episodes < n:
            obs, rew, done, info = env.step(act)
            cum_rew += rew
            act = mountain_car_func(obs, action_mapping)
            if np.any(done):
                episodes += np.sum(done)
                episode_rewards += list(cum_rew[done])
                cum_rew[done] = 0
        print(f"\tGoal Position:{gp}\tMean Reward:{np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    # mountain_car_goal_test()
    # exit(0)
    is_valid = False
    n_envs = 2

    # logdir = "logs/train/cartpole_swing/test/2024-05-01__14-19-53__seed_6033/"
    # policy, _, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs)
    # env = CartPoleVecEnv(n_envs, degrees=12, h_range=2.4, max_steps=500, render_mode="human")
    # env = AcrobotVecEnv(n_envs)
    env = MountainCarVecEnv(n_envs=n_envs, goal_position=0.5, render_mode="human", sparse_rewards=True)
    # env = get_env_constructor("cartpole_swing")(DictToArgs({"render": True, "seed":6033}), {}, is_valid)
    action_mapping = map_actions_to_values(get_actions_from_all(env))

    # cust_model = CustomModel()
    # s_agent = symbolic_agent_constructor(cust_model, policy, False, action_mapping)

    obs = env.reset()
    cum_rew = []
    while True:
        # act = cartpole_swing_func(obs, action_mapping)
        act = mountain_car_func(obs, action_mapping)
        # act = s_agent.forward(obs)
        ep_rew = env.n_steps[0]
        obs, rew, done, info = env.step(act)
        cum_rew += [rew[0]]
        rewards = []
        if done[0]:
            episode_reward = np.sum(cum_rew)
            print(f"Episode Reward: {episode_reward :.2f}")
            obs = env.reset(seed=np.random.randint(0, 5000))
            cum_rew = []
            rewards.append(episode_reward)
