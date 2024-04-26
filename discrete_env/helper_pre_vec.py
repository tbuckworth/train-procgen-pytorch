import numpy as np


class StartSpace:
    def __init__(self, low, high, np_random):
        assert len(low) == len(high), "low and high must be the same length"
        self.low = np.array(low)
        self.high = np.array(high)
        self._np_random = np_random
        self.n_inputs = len(low)

    def sample(self, n):
        return self._np_random.uniform(low=self.low, high=self.high, size=(n, self.n_inputs))

    def set_np_random(self, np_random):
        self._np_random = np_random


def override_value(env_args, hyperparameters, suffix, param, value):
    env_args[param] = hyperparameters.get(f"{param}{suffix}", value)


def assign_env_vars(hyperparameters, is_valid, overrides):
    env_args = {}
    suffix = ""
    if is_valid:
        suffix = "_v"
    for k, v in overrides.items():
        override_value(env_args, hyperparameters, suffix, k, v[0])
    return env_args
