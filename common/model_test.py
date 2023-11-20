import unittest

import numpy as np
import torch
from torchinfo import summary

from common.model import ImpalaVQMHAModel
from helper import create_env


class MyTestCase(unittest.TestCase):
    def test_ImpalaVQMHAModel(self):
        device = torch.device('cpu')
        env_args = {"num": 2,
                    "env_name": "coinrun",
                    "start_level": 325,
                    "num_levels": 1,
                    "paint_vel_info": True,
                    "distribution_mode": "hard"}
        env = create_env(env_args, render=False, normalize_rew=True)
        in_channels = env.observation_space.shape[0]
        obs = env.reset()
        model = ImpalaVQMHAModel(in_channels, 1, device)
        obs = torch.FloatTensor(obs)
        model.forward(obs)
        summary(model, obs.shape)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
