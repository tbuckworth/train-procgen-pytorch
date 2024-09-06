import gymnasium as gym

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
        return obs, rew, done, infos

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)[0]


if __name__ == "__main__":

    env = GymnasiumEnv('Humanoid-v4', 2)
    obs = env.reset(seed=42)
    for _ in range(1000):
        actions = env.env.action_space.sample()
        obs, rew, done, info = env.step(actions)


    print(obs.shape)
