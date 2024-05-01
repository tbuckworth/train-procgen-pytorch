import numpy as np


class StartSpace:
    def __init__(self, low, high, np_random, use_loc=False):
        assert len(low) == len(high), "low and high must be the same length"
        self.low = np.array(low)
        self.high = np.array(high)
        self._np_random = np_random
        self.n_inputs = len(low)
        self.use_loc = use_loc

    def sample(self, n):
        if self.use_loc:
            return self._np_random.normal(loc=self.low, scale=self.high, size=(n, self.n_inputs))
        return self._np_random.uniform(low=self.low, high=self.high, size=(n, self.n_inputs))

    def set_np_random(self, np_random):
        self._np_random = np_random


def override_value(env_args, hyperparameters, suffix, param, value):
    env_args[param] = hyperparameters.get(f"{param}{suffix}", value)


def assign_env_vars(hyperparameters, is_valid, overrides):
    env_args = {}
    suffix = ""
    i = 0
    if is_valid:
        suffix = "_v"
        i = -1
    for k, v in overrides.items():
        if is_valid and k == "n_envs":
            override_value(env_args, hyperparameters, "", k, v[i])
        override_value(env_args, hyperparameters, suffix, k, v[i])
    return env_args
