import unittest

import numpy as np
import torch

from common.model import ImpalaVQMHAModel
from helper import create_env


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # image_file = r"..\logs\train\coinrun\coinrun\2023-10-31__10-49-30__seed_6033\go_left.npy"
        # frames = np.load(image_file, allow_pickle='TRUE')

        env_args = {"num": 2,
                    "env_name": "coinrun",
                    "start_level": 325,
                    "num_levels": 1,
                    "paint_vel_info": True,
                    "distribution_mode": "hard"}
        env = create_env(env_args, render=False, normalize_rew=True)
        in_channels = env.observation_space.shape[0]
        obs = env.reset()
        model = ImpalaVQMHAModel(in_channels)
        # with torch.no_grad():
        #     obs = torch.FloatTensor(obs)
        #     model.forward(obs)
        obs = torch.FloatTensor(obs)
        model.forward(obs)

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
