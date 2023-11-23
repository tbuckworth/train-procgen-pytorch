import unittest

import numpy as np
import torch
from torchinfo import summary

from common.model import ImpalaVQMHAModel, ImpalaFSQModel
from helper import create_env, initialize_model


class MyTestCase(unittest.TestCase):
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
        model = ImpalaVQMHAModel(self.in_channels, 1, self.device, use_vq=False)
        model.forward(self.obs, print_nans=True)
        summary(model, self.obs.shape)
        hyperparameters = {"architecture": "impalavqmha",
                           "mha_layers": 2,
                           "use_vq": False,
                           }
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        policy.forward(self.obs, None, None)

    def test_ImpalaFSQModel(self):
        model = ImpalaFSQModel(self.in_channels)
        model.forward(self.obs)
        summary(model, self.obs.shape)

if __name__ == '__main__':
    unittest.main()
