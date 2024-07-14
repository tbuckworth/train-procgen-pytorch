import os
import unittest
from unittest.mock import Mock

import numpy as np


from cartpole.create_cartpole import create_cartpole
from common.env.env_constructor import get_env_constructor
from common.env.procgen_wrappers import create_procgen_env
from discrete_env.acrobot_pre_vec import create_acrobot
from discrete_env.mountain_car_pre_vec import create_mountain_car

from helper_local import DictToArgs, initialize_model, get_actions_from_all, \
    map_actions_to_values, get_config, load_hparams_for_model

from symbreg.agents import SymbolicAgent, NeuroSymbolicAgent, DoubleGraphSymbolicAgent
import torch
from torch import device as torch_device

class MockRegressor:
    def __init__(self, n_out):
        self.n_out = n_out

    def predict(self, x):
        if self.n_out == 1:
            return np.random.random((x.shape[0],))
        return np.random.random((x.shape[0], self.n_out))

class BaseAgentTester(unittest.TestCase):
    agent = None
    env = None
    model = None
    arch = None
    symbolic_agent_constructor = None
    action_mapping = None

    def setUp(cls):
        device = torch_device("cpu")
        _, _, policy = initialize_model(device, cls.env, {"architecture": cls.arch})
        actions = get_actions_from_all(cls.env)
        cls.action_mapping = map_actions_to_values(actions)
        cls.stoch_model = MockRegressor(cls.env.action_space.n)
        cls.det_model = MockRegressor(1)
        cls.agent = cls.symbolic_agent_constructor(None, policy, stochastic=False, action_mapping=cls.action_mapping)

        cls.obs = cls.env.reset()

    def run_all(self):
        self.forward_sample(stochastic=False)
        self.forward_sample(stochastic=True)

    def forward_sample(self, stochastic):
        self.agent.stochastic = stochastic
        if stochastic and not self.agent.single_output:
            self.agent.model = self.stoch_model
        else:
            self.agent.model = self.det_model
        x, y, act, value = self.agent.sample(self.obs)
        act_hat = self.agent.forward(self.obs)
        self.assertEqual(act_hat.shape, act.shape)


class TestSymbolicMountainCar(BaseAgentTester):
    def setUp(cls):
        cls.arch = "mlpmodel"
        cls.env = create_mountain_car(None, {})
        cls.symbolic_agent_constructor = SymbolicAgent
        super(TestSymbolicMountainCar, cls).setUp()

    def tests(self):
        self.run_all()


class TestSymbolicCartpole(BaseAgentTester):
    def setUp(cls):
        cls.arch = "mlpmodel"
        cls.env = create_cartpole(None, {})
        cls.symbolic_agent_constructor = SymbolicAgent
        super(TestSymbolicCartpole, cls).setUp()

    def tests(self):
        self.run_all()

class TestSymbolicAcrobot(BaseAgentTester):
    def setUp(cls):
        cls.arch = "mlpmodel"
        cls.env = create_acrobot(None, {})
        cls.symbolic_agent_constructor = SymbolicAgent
        super(TestSymbolicAcrobot, cls).setUp()

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


class TestDoubleGraphAgent(unittest.TestCase):
    def setUp(cls):
        cls.env_cons = get_env_constructor("cartpole")
        cls.symbolic_agent_constructor = DoubleGraphSymbolicAgent
        cls.logdir = "logs/train/cartpole/2024-07-11__04-48-25__seed_6033"
        cls.data_size = 1
    def tests(self):
        cfg = get_config(self.logdir)

        hyperparameters, last_model = load_hparams_for_model(cfg["param_name"], self.logdir, n_envs=2)
        env = self.env_cons(None, hyperparameters)
        device = torch.device("cpu")
        model, observation_shape, policy = initialize_model(device, env, hyperparameters)

        nn_agent = self.symbolic_agent_constructor(policy)
        obs = env.reset()
        nn_agent.forward(obs)
        m_in, m_out, u_in, u_out, vm_in, vm_out, vu_in, vu_out = nn_agent.sample(obs)

if __name__ == '__main__':
    unittest.main()
