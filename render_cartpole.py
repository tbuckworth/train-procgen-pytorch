import numpy as np

from cartpole.cartpole_pre_vec import CartPoleVecEnv


def symbolic_regression_function(obs):
    x0, x1, x2, x3 = obs[0]
    if x0 > -1.62 * (3 * x2 + x3 + x1):
        return np.ones(len(obs))
    return np.zeros(len(obs))

def alt_func(obs):
    x0, x1, x2, x3 = obs.T
    return (0.57+5*x2+x3).round(decimals=0)

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

    env = CartPoleVecEnv(n_envs, degrees=9, h_range=1.8, max_steps=500, render_mode="human")
    obs = env.reset()
    while True:
        act = alt_func(obs)
        ep_rew = env.n_steps[0]
        obs, rew, done, info = env.step(act)
        if done[0]:
            print("Episode finished after {} timesteps".format(ep_rew))
            obs = env.reset(seed=np.random.randint(0,5000))
