import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cartpole.cartpole_pre_vec import CartPoleVecEnv
from helper_local import DictToArgs

def create_mujoco(env_name):
    def mujoco_cons(args, hyperparameters, is_valid=False):
        if args is None:
            args = DictToArgs({"render": False, "seed": 0})
        n_envs = hyperparameters.get('n_envs', 8)
        timeout = hyperparameters.get('env_timeout', 10)
        env = GymnasiumEnv(env_name, n_envs=n_envs, timeout=timeout, render=args.render)
        if is_valid:
            env.reset(seed=args.seed + 1000)
        else:
            env.reset(seed=args.seed)
        return env
    return mujoco_cons


def remove_batch_from_space(space):
    if len(space.shape) == 1:
        return space
    high = space.high[0]
    low = space.low[0]
    shape = space.shape[1:]
    dtype = space.dtype
    return spaces.Box(low, high, shape, dtype)


class GymnasiumEnv(gym.Env):
    def __init__(self, env_name, n_envs, render=False, timeout=10):
        render_mode = None
        if render:
            render_mode = "human"
        self.env = gym.vector.make(env_name, num_envs=n_envs, asynchronous=True, render_mode=render_mode)
        self.action_space = remove_batch_from_space(self.env.action_space)
        self.observation_space = remove_batch_from_space(self.env.observation_space)
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
    env_name = "Ant-v4"
    env = gym.vector.make(env_name, num_envs=2, asynchronous=True, render_mode=None)
    env.reset()
    env.step(env.action_space.sample())

    cenv = CartPoleVecEnv(2)
    env = GymnasiumEnv(env_name, 2)
    obs = env.reset(seed=42)
    actions = (np.random.random((2, env.action_space.shape[0])) - 0.5)*0.4
    obs, rew, done, info = env.step(actions)
    cenv.reset()
    actions = np.array([cenv.action_space.sample() for _ in range(2)])
    obs, rew, done, c_info = cenv.step(actions)


    print(obs.shape)
