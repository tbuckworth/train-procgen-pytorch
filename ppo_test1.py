import unittest
import torch
import os
from agents.ppo import PPO
from agents.ppo_model import PPOModel
from common.env.env_constructor import get_env_constructor
from common.env.procgen_wrappers import create_env
from common.logger import Logger
from common.storage import Storage, BasicStorage
from discrete_env.mountain_car_pre_vec import MountainCarVecEnv
from helper_local import initialize_model, get_hyperparams


class TestPPO(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        print(os.getcwd())
        n_envs = 2
        cls.device = torch.device('cpu')
        env_args = {"num": n_envs,
                    "env_name": "coinrun",
                    "start_level": 325,
                    "num_levels": 1,
                    "paint_vel_info": True,
                    "distribution_mode": "hard"}
        cls.env = create_env(env_args, render=False, normalize_rew=True)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("hard-500-impala")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False)
        logger.max_steps = 1000

        hidden_state_dim = model.output_dim
        storage = Storage(cls.obs_shape, hidden_state_dim, cls.n_steps, n_envs, cls.device)

        cls.agent = PPO(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_ppo(self):
        self.agent.train(int(1e5))

    def test_mountain_car_pre_vec(self):
        n_envs = 2
        env = MountainCarVecEnv(n_envs=n_envs)
        env.reset()

        hyperparameters = {"n_envs": n_envs, "architecture": "mlpmodel", "n_steps": self.n_steps, "lmbda":0.98}
        model, obs_shape, policy = initialize_model(self.device, env, hyperparameters)
        logger = Logger(hyperparameters.get("n_envs"), self.logdir, use_wandb=False, has_vq=False)
        logger.max_steps = 1000

        hidden_state_dim = model.output_dim
        storage = Storage(obs_shape, hidden_state_dim, self.n_steps, hyperparameters.get("n_envs"), self.device)

        agent = PPO(env, policy, logger, storage, self.device,1, **hyperparameters)
        agent.train(int(1e5))



class TestPPOModel(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = torch.device('cpu')
        env_con = get_env_constructor("cartpole_swing")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("graph-transition")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        hyperparameters["anneal_temp"] = True
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, transition_model=True)
        logger.max_steps = 500

        storage = BasicStorage(cls.obs_shape, cls.n_steps, n_envs, cls.device)

        cls.agent = PPOModel(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_ppo(self):
        self.agent.train(int(1e5))





if __name__ == '__main__':
    unittest.main()
