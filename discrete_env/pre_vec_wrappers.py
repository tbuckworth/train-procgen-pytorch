import gymnasium as gym


class PetsWrapper(gym.Env):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, False, info

    def reset(self, seed=None):
        return self.env.reset(seed), {}

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class DeVecEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.n_envs = self.env.n_envs

    def step(self, action):
        shp = [self.n_envs] + [1 for _ in action.shape]
        action = action.tile(shp)
        obs, rew, done, info = self.env.step(action)
        return obs[0], rew[0], done[0], info[0]

    def reset(self, seed=None):
        return self.env.reset(seed)[0]

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
