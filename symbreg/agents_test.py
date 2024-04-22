import os
import unittest
from unittest.mock import Mock

import numpy as np
from torch import device as torch_device

from common.env.procgen_wrappers import create_procgen_env
from discrete_env.mountain_car_pre_vec import create_mountain_car
from helper_local import get_config, DictToArgs, get_path, initialize_model
from symbolic_regression import load_nn_policy, generate_data
from symbreg.agents import SymbolicAgent, NeuroSymbolicAgent
from pysr import PySRRegressor


class MockRegressor:
    def __init__(self, n_out):
        self.n_out = n_out

    def predict(self, x):
        return np.random.random((x.shape[0], self.n_out))


# class MyTestCase(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         n_envs = 32
#         cls.data_size = 100
#         logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
#         cls.policy, cls.env, cls.symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs)
#         cls.agent = cls.symbolic_agent_constructor(None, cls.policy, stochastic=False, action_mapping=None)
#         X, Y, V = generate_data(cls.agent, cls.env, int(cls.data_size))
#
#         symbdir = os.path.join(logdir, "symbreg/2024-04-12__17-38-41/")
#         pickle_filename = get_path(symbdir, "symb_reg.pkl")
#
#         cls.model = PySRRegressor.from_file(pickle_filename)
#
#     def test_deterministic_symbolic_agent_sample(self):
#         self.agent.stochastic = False
#         x, y, v = generate_data(self.agent, self.env, int(self.data_size))
#         y_hat = self.agent.forward(x)
#         # self.assertEqual(Y_hat.shape[], Y.shape)
#
#     def test_stochastic_symbolic_agent_sample(self):
#         self.agent.stochastic = True
#         x, y, v = generate_data(self.agent, self.env, int(self.data_size))
#         y_hat = self.agent.forward(x)
#         self.assertEqual(y_hat.shape, y.shape)


class BaseAgentTester(unittest.TestCase):
    agent = None
    env = None
    model = None
    arch = None
    symbolic_agent_constructor = None

    def setUp(cls):
        device = torch_device("cpu")
        # cls.env = create_mountain_car(None, {})
        _, _, policy = initialize_model(device, cls.env, {"architecture": cls.arch})
        cls.model = MockRegressor(cls.env.action_space.n)
        cls.agent = cls.symbolic_agent_constructor(cls.model, policy, stochastic=False, action_mapping=None)

        cls.obs = cls.env.reset()

    def run_all(self):
        self.forward_sample(stochastic=False)
        self.forward_sample(stochastic=True)

    def forward_sample(self, stochastic):
        self.agent.stochastic = stochastic
        x, y, act, value = self.agent.sample(self.obs)
        act_hat = self.agent.forward(x)
        self.assertEqual(act_hat.shape, act.shape)


class TestSymbolicMountainCar(BaseAgentTester):
    def setUp(cls):
        cls.arch = "mlpmodel"
        cls.env = create_mountain_car(None, {})
        cls.symbolic_agent_constructor = SymbolicAgent
        super(TestSymbolicMountainCar, cls).setUp()

    def tests(self):
        self.run_all()


class TestNeuroSymbolicCoinrun(BaseAgentTester):
    def setUp(cls):
        cls.arch = "impala"

        args = DictToArgs({
            "device": "cpu",
            "real_procgen": True,
            "val_env_name": "coinrun",
            "start_level": 0,
            "num_levels": 500,
            "env_name": "coinrun",
            "distribution_mode": "hard",
            "num_threads": 8,
            "architecture": cls.arch,
            "reduce_duplicate_actions": True,
        })
        hyperparameters = {"n_envs": 2}
        cls.env = create_procgen_env(args, hyperparameters)
        cls.symbolic_agent_constructor = NeuroSymbolicAgent
        super(TestNeuroSymbolicCoinrun, cls).setUp()

    def tests(self):
        self.run_all()

    #TODO: test coinrun


if __name__ == '__main__':
    unittest.main()
