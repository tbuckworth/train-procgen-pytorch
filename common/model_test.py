import time
import unittest

import numpy as np
import torch
import yaml
from torchinfo import summary

from boxworld.create_box_world import create_box_world_env_pre_vec
from cartpole.create_cartpole import create_cartpole_env_pre_vec
from common.env.env_constructor import get_env_constructor
from common.model import ImpalaVQMHAModel, ImpalaFSQModel, ImpalaModel, Decoder, VQVAE, ImpalaFSQMHAModel, \
    RibFSQMHAModel
from common.storage import Storage
from helper_local import initialize_model, get_config, get_saved_hyperparams, load_hparams_for_model
from common.env.procgen_wrappers import create_env
from inspect_agent import predict


def get_hyperparams(param_name):
    with open('../hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    return hyperparameters


class CoinrunTestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device('cpu')
        cls.n_envs = 128
        env_args = {"num": cls.n_envs,
                    "env_name": "coinrun",
                    "start_level": 0,
                    "num_levels": 500,
                    "paint_vel_info": True,
                    "distribution_mode": "hard"}
        cls.env = create_env(env_args, render=False, normalize_rew=True, reduce_duplicate_actions=True)
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
        x1 = model.forward(self.obs)
        summary(model, self.obs.shape)

        feature_batch, atn_batch_list, feature_indices, _ = model.forward_with_attn_indices(
            self.obs)
        print("x")

    def test_ImpalaModel(self):
        model = ImpalaModel(self.in_channels)

        model.forward_with_attn_indices(self.obs)
        summary(model, self.obs.shape)

    def test_ImpalaModelPolicyLoaded(self):
        logdir = "logs/train/coinrun/coinrun-hparams/2024-04-18__08-38-17__seed_6033"
        cfg = get_config(logdir)
        hyperparameters, last_model = load_hparams_for_model(cfg["param_name"], logdir, self.n_envs)

        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        checkpoint = torch.load(last_model, map_location=self.device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.embedder.forward_with_attn_indices(self.obs)
        # policy.forward(self.obs, None, None)


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

    def test_ImpalaMHA(self):
        hyperparameters = get_hyperparams("hard-500-impalamha")
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        model.forward(self.obs)
        summary(model, self.obs.shape)
        policy.forward(self.obs, None, None)
        feature_batch, atn_batch_list, feature_indices, _ = model.forward_with_attn_indices(
            self.obs)

    def test_ImpalaITN(self):
        hyperparameters = get_hyperparams("hard-500-impalaitn")
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        model.forward(self.obs)
        summary(model, self.obs.shape)
        policy.forward(self.obs, None, None)
        feature_batch, atn_batch_list, feature_indices, _ = model.forward_with_attn_indices(
            self.obs)

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

class CartPoleTestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device('cpu')
        env_args = {"n_envs": 10}
        cls.n_envs = env_args["n_envs"]
        env_cons = get_env_constructor("cartpole")
        cls.env = env_cons(None, env_args, False)
        # cls.env = create_cartpole_env_pre_vec(env_args, render=False, normalize_rew=True)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

    def test_MLPModel(self):
        # model = RibFSQMHAModel(self.in_channels, 2, self.device, self.obs_shape, reduce='dim_wise')
        hyperparameters = get_hyperparams("cartpole")
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        model.forward(self.obs)
        summary(model, self.obs.shape)

        policy.device = self.device
        hidden_state_dim = model.output_dim
        storage = Storage(self.obs_shape, hidden_state_dim, 256, self.n_envs, self.device)
        hidden_state = np.zeros((self.n_envs, storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        act, log_prob_act, value, next_hidden_state, pi = predict(policy, self.obs, hidden_state, done)
        next_obs, rew, done, info = self.env.step(act)
        storage.store(next_obs, hidden_state, act, rew, done, info, log_prob_act, value)
        rew_batch, done_batch, true_average_reward = storage.fetch_log_data()

    def test_TransformoBot(self):
        hyperparameters = get_hyperparams("cartpole_transform")
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        model.forward(self.obs)
        summary(model, self.obs.shape)

    def test_graph_transition_model(self):
        hyperparameters = get_hyperparams("graph-transition")
        model, obs_shape, policy = initialize_model(self.device, self.env, hyperparameters)
        action = torch.FloatTensor([self.env.action_space.sample() for _ in range(self.n_envs)])
        # model.forward(self.obs)
        # summary(model, self.obs.shape)
        policy.forward(self.obs)
        n = 100
        start = time.time()
        for _ in range(n):
            policy.transition_model.forward(self.obs, policy.actions_like(self.obs, 0))
        mid = time.time()
        for _ in range(n):
            policy.transition_model.old_forward(self.obs, policy.actions_like(self.obs, 0))
        end = time.time()
        print(f"Vectorized:\t{mid-start:.4f}")
        print(f"Looped:\t{end-mid:.4f}")
        print(f"Ratio:\t{(end-mid)/(mid-start):.2f}")

if __name__ == '__main__':
    unittest.main()
