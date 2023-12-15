import unittest

import numpy as np
import torch
from torchinfo import summary

from boxworld.box_world_env import create_box_world_env
from common.model import ImpalaVQMHAModel, ImpalaFSQModel
from helper import initialize_model
from common.env.procgen_wrappers import create_env


class CoinrunTestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device('cpu')
        env_args = {"num": 2,
                    "env_name": "coinrun",
                    "start_level": 325,
                    "num_levels": 1,
                    "paint_vel_info": True,
                    "distribution_mode": "hard"}
        cls.env = create_env(env_args, render=False, normalize_rew=True)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())

    def test_ImpalaVQMHAModel(self):
        model = ImpalaVQMHAModel(self.in_channels, 1, self.device, use_vq=True)
        model.forward(self.obs)
        summary(model, self.obs.shape)
        hyperparameters = {"architecture": "impalavqmha",
                           "mha_layers": 2,
                           "use_vq": True,
                           }
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        policy.forward(self.obs, None, None)

    def test_ImpalaFSQModel(self):
        model = ImpalaFSQModel(self.in_channels, self.device, use_mha=True)
        model.forward(self.obs)
        summary(model, self.obs.shape)


class BoxWorldTestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device('cpu')
        env_args = {"num": 2,
                    "env_name": "BoxWorld-v0",
                    "n": 12,
                    "goal_length": 5,
                    "num_distractor": 3,
                    "distractor_length": 3,
                    }
        cls.env = create_box_world_env(env_args, render=False, normalize_rew=True)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())

    def test_ImpalaVQMHAModel(self):
        model = ImpalaVQMHAModel(self.in_channels, 1, self.device, use_vq=True)
        model.forward(self.obs)
        summary(model, self.obs.shape)
        hyperparameters = {"architecture": "impalavqmha",
                           "mha_layers": 2,
                           "use_vq": True,
                           }
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        policy.forward(self.obs, None, None)


if __name__ == '__main__':
    unittest.main()
