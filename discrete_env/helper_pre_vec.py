import numpy as np

from boxworld.create_box_world import create_bw_env
from cartpole.create_cartpole import create_cartpole
from common.env.procgen_wrappers import create_procgen_env
from discrete_env.acrobot_pre_vec import create_acrobot
from discrete_env.mountain_car_pre_vec import create_mountain_car


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


def get_env_constructor(env_name):
    create_venv = create_procgen_env
    if env_name == "boxworld":
        create_venv = create_bw_env
    if env_name == "cartpole":
        create_venv = create_cartpole
    if env_name == "mountain_car":
        create_venv = create_mountain_car
    if env_name == "acrobot":
        create_venv = create_acrobot
    if env_name == "lunar_lander":
        raise NotImplementedError
    return create_venv
