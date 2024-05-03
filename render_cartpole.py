import numpy as np

from cartpole.cartpole_pre_vec import CartPoleVecEnv
from common.env.env_constructor import get_env_constructor
from discrete_env.acrobot_pre_vec import AcrobotVecEnv
from helper_local import sigmoid, sample_from_sigmoid, DictToArgs, map_actions_to_values, get_actions_from_all, match, \
    match_to_nearest


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

if __name__ == "__main__":
    is_valid = False
    n_envs = 2

    # env = CartPoleVecEnv(n_envs, degrees=12, h_range=2.4, max_steps=500, render_mode="human")
    # env = AcrobotVecEnv(n_envs)
    env = get_env_constructor("cartpole_swing")(DictToArgs({"render": True, "seed":6033}), {}, is_valid)
    action_mapping = map_actions_to_values(get_actions_from_all(env))

    obs = env.reset()
    cum_rew = []
    while True:
        act = cartpole_swing_func(obs, action_mapping)
        ep_rew = env.n_steps[0]
        obs, rew, done, info = env.step(act)
        cum_rew += [rew[0]]
        if done[0]:
            print(f"Episode Reward: {np.sum(cum_rew):.2f}")
            obs = env.reset(seed=np.random.randint(0, 5000))
            cum_rew = []
