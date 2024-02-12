import unittest

import numpy as np
import torch
import yaml
from torchinfo import summary

from boxworld.create_box_world import create_box_world_env, create_box_world_env_pre_vec
from common.model import ImpalaVQMHAModel, ImpalaFSQModel, ImpalaModel, Decoder, VQVAE, ImpalaFSQMHAModel, \
    RibFSQMHAModel
from helper import initialize_model
from common.env.procgen_wrappers import create_env


def get_hyperparams(param_name):
    with open('../hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    return hyperparameters


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
        cls.obs_shape = cls.env.observation_space.shape


    def testVQVAEModel(self):
        hyperparameters = get_hyperparams("vq-vae")

        model = VQVAE(self.in_channels, **hyperparameters)
        model.to(self.device)
        model.forward(self.obs)
        summary(model, self.obs.shape)

    def test_ImpalaVQMHAModel(self):
        obs_shape = self.env.observation_space.shape
        model = ImpalaVQMHAModel(self.in_channels, 1, self.device, obs_shape, use_vq=True)
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

    def test_ImpalaFSQMHAModel(self):
        model = ImpalaFSQMHAModel(self.in_channels, 2, self.device, self.obs_shape, reduce='dim_wise')
        model.forward(self.obs)
        summary(model, self.obs.shape)

    def test_ImpalaModel(self):
        model = ImpalaModel(self.in_channels)
        model.forward(self.obs)
        summary(model, self.obs.shape)

    def test_ImpalaDecoder(self):
        model = ImpalaModel(self.in_channels)
        x = model.forward(self.obs)
        print(f"Input shape: {self.obs.shape}")
        summary(model, self.obs.shape)
        hyperparameters = get_hyperparams("hard-500-impala")
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        policy.forward(self.obs, None, None)

        x = model.block1(self.obs)
        x = model.block2(x)
        x = model.block3(x)
        print(x.shape)
        x = model.forward(self.obs)
        print(x.shape)
        x = x.reshape(2, 1, 16, 16)
        decoder = Decoder(
            embedding_dim=1,
            num_hiddens=64,
            num_upsampling_layers=2,
            num_residual_layers=2,
            num_residual_hiddens=32,
        )

        decoder.forward(x)
        summary(decoder, x.shape)


class BoxWorldTestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device('cpu')
        env_args = {"n_envs": 2,
                    # "env_name": "BoxWorld-v0",
                    "n": 6,
                    "goal_length": 3,
                    "num_distractor": 1,
                    "distractor_length": 2,
                    "max_steps": 10 ** 6,
                    "seed": 0,
                    "n_levels": 0
                    }
        cls.env = create_box_world_env_pre_vec(env_args, render=False, normalize_rew=True)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape


    def test_ImpalaVQMHAModel(self):
        hyperparameters = {"architecture": "impalavqmha",
                           "mha_layers": 2,
                           "use_vq": True,
                           }
        # num_heads must be 1 for 6x6 puzzle
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        model.forward(self.obs)
        summary(model, self.obs.shape)

        policy.forward(self.obs, None, None)



    def test_ribMHA(self):
        hyperparameters = {"architecture": "ribmha",
                           "use_vq": False
                           }
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        model.forward(self.obs)
        summary(model, self.obs.shape)
        policy.forward(self.obs, None, None)

    def test_ribVQMHA(self):
        hyperparameters = {"architecture": "ribmha",
                           "use_vq": True
                           }
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        model.forward(self.obs)
        summary(model, self.obs.shape)
        policy.forward(self.obs, None, None)

    def test_RibFSQMHAModel(self):
        # model = RibFSQMHAModel(self.in_channels, 2, self.device, self.obs_shape, reduce='dim_wise')
        hyperparameters = get_hyperparams("boxworld-ribfsqmha-easy")
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        model.forward(self.obs)
        summary(model, self.obs.shape)

if __name__ == '__main__':
    unittest.main()
