import numpy as np

from cartpole.cartpole_pre_vec import CartPoleVecEnv


def symbolic_regression_function(obs):
    x0, x1, x2, x3 = obs[0]
    if x0 > -1.62 * (3 * x2 + x3 + x1):
        return np.ones(len(obs))
    return np.zeros(len(obs))


if __name__ == "__main__":
    is_valid = False
    n_envs = 2

    env = CartPoleVecEnv(n_envs, degrees=9, h_range=1.8, max_steps=500, render_mode="human")
    obs = env.reset()
    while True:
        act = symbolic_regression_function(obs)
        obs, rew, done, info = env.step(act)
