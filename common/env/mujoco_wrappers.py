import gymnasium as gym
import numpy as np

from cartpole.cartpole_pre_vec import CartPoleVecEnv
from helper_local import DictToArgs


def create_humanoid(args, hyperparameters, is_valid=False):
    if args is None:
        args = DictToArgs({"render": False, "seed": 0})
    n_envs = hyperparameters.get('n_envs', 8)
    timeout = hyperparameters.get('env_timeout', 10)
    env = GymnasiumEnv('Humanoid-v4', n_envs=n_envs, timeout=timeout)
    if is_valid:
        env.reset(seed=args.seed + 1000)
    else:
        env.reset(seed=args.seed)
    return env


class GymnasiumEnv(gym.Env):
    def __init__(self, env_name, n_envs, timeout=10):
        self.env = gym.vector.make(env_name, num_envs=n_envs, asynchronous=False)
        self.action_space = self.env.action_space
        self.action_space._shape = self.env.action_space.shape[1:]
        self.observation_space = self.env.observation_space
        self.observation_space._shape = self.env.observation_space.shape[1:]

        self.timeout = timeout

    def step(self, actions):
        # self.env.step_async(actions)
        # obs, rew, done, trunc, infos = self.env.step_wait(timeout=None)
        obs, rew, done, trunc, infos = self.env.step(actions)
        done |= trunc
        info = [{k:v[i] for k, v in infos.items()} for i in range(len(obs))]
        return obs, rew, done, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)[0]


if __name__ == "__main__":
    cenv = CartPoleVecEnv(2)
    env = GymnasiumEnv('Humanoid-v4', 2)
    obs = env.reset(seed=42)
    actions = (np.random.random((2,17)) - 0.5)*0.4
    obs, rew, done, info = env.step(actions)
    cenv.reset()
    actions = np.array([cenv.action_space.sample() for _ in range(2)])
    obs, rew, done, c_info = cenv.step(actions)


    print(obs.shape)
