import unittest
from symbol import if_stmt

import torch
import os

from agents.IPL import IPL
from agents.IPL_ICM import IPL_ICM
from agents.double_graph_agent import DoubleGraphAgent
from agents.espo import ESPO
from agents.goal_seeker import GoalSeeker
from agents.graph_agent import GraphAgent
from agents.ppo import PPO
from agents.ppo_model import PPOModel
from agents.ppo_pure import PPOPure
from agents.ppp_model import PPPModel
from common.env.env_constructor import get_env_constructor
from common.env.procgen_wrappers import create_env
from common.logger import Logger
from common.storage import Storage, BasicStorage, IPLStorage, GoalSeekerStorage
from discrete_env.mountain_car_pre_vec import MountainCarVecEnv
from helper_local import initialize_model, get_hyperparams, initialize_storage


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
        env_con = get_env_constructor("cartpole")
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
        logger.max_steps = 1000

        storage = BasicStorage(cls.obs_shape, cls.n_steps, n_envs, cls.device)

        cls.agent = PPOModel(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_ppomodel(self):
        self.agent.train(int(1e5))



class TestPPPModel(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = torch.device('cpu')
        env_con = get_env_constructor("coinrun")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("pixel-graph-transition")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        hyperparameters["anneal_temp"] = True
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, transition_model=True)
        logger.max_steps = 1000

        storage = BasicStorage(cls.obs_shape, cls.n_steps, n_envs, cls.device)

        cls.agent = PPPModel(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_ppp(self):
        self.agent.train(int(1e5))



class TestGraphAgent(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = torch.device('cpu')
        env_con = get_env_constructor("cartpole")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("full-graph-transition")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        hyperparameters["anneal_temp"] = True
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, transition_model=True)
        logger.max_steps = 1000

        storage = BasicStorage(cls.obs_shape, cls.n_steps, n_envs, cls.device)

        cls.agent = GraphAgent(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_graph_agent(self):
        self.agent.train(int(1e5))

class TestDoubleGraphAgent(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = torch.device('cpu')
        env_con = get_env_constructor("cartpole")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("double-graph")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        hyperparameters["anneal_temp"] = False
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, double_graph=True)
        logger.max_steps = 1000

        storage = BasicStorage(cls.obs_shape, cls.n_steps, n_envs, cls.device)

        cls.agent = DoubleGraphAgent(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_double_graph_agent(self):
        self.agent.train(int(1e5))




class TestPPOPure(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = torch.device('cpu')
        env_con = get_env_constructor("cartpole_continuous")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("graph-cartpole-cont")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        # hyperparameters["anneal_temp"] = False
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, ppo_pure=True)
        logger.max_steps = 1000

        act_shape = policy.act_shape
        hidden_state_dim = model.output_dim
        storage = Storage(cls.obs_shape, hidden_state_dim, cls.n_steps, n_envs, cls.device, continuous_actions=True, act_shape=act_shape)

        cls.agent = PPOPure(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_ppo_pure(self):
        self.agent.train(int(1e5))


class TestESPO(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = torch.device('cpu')
        env_con = get_env_constructor("cartpole_continuous")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("eql-graph")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        # hyperparameters["anneal_temp"] = False
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, ppo_pure=True)
        logger.max_steps = 1000

        act_shape = policy.act_shape
        hidden_state_dim = model.output_dim
        storage = Storage(cls.obs_shape, hidden_state_dim, cls.n_steps, n_envs, cls.device, continuous_actions=True, act_shape=act_shape)

        cls.agent = ESPO(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_espo(self):
        self.agent.train(int(1e5))

class TestIPL(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        env_con = get_env_constructor("cartpole")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("ipl_cartpole")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        # hyperparameters["anneal_temp"] = False
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, algo=hyperparameters["algo"])
        logger.max_steps = 1000

        storage = IPLStorage(cls.obs_shape, cls.n_steps, n_envs, cls.device)

        cls.agent = IPL(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_ipl(self):
        self.agent.train(int(1e5))

class TestIPL_ICM(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        env_con = get_env_constructor("cartpole")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("ipl_icm_cartpole")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        # hyperparameters["anneal_temp"] = False
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, IPL=True)
        logger.max_steps = 1000

        storage = IPLStorage(cls.obs_shape, cls.n_steps, n_envs, cls.device)

        cls.agent = IPL_ICM(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_ipl_icm(self):
        self.agent.train(int(1e5))

class TestGoalSeeker(unittest.TestCase):
    device = None
    obs_shape = None
    env = None

    @classmethod
    def setUpClass(cls):
        n_envs = 2
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        env_con = get_env_constructor("cartpole")
        hyperparameters = {"n_envs": n_envs}
        cls.env = env_con(None, hyperparameters)
        cls.in_channels = cls.env.observation_space.shape[0]
        cls.obs = torch.FloatTensor(cls.env.reset())
        cls.obs_shape = cls.env.observation_space.shape

        logdir = "logs/test/test"
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        cls.logdir = logdir
        hyperparameters = get_hyperparams("goal-seeker-mlp")
        cls.n_steps = hyperparameters.get("n_steps", 256)
        hyperparameters["n_envs"] = n_envs
        # hyperparameters["anneal_temp"] = False
        model, obs_shape, policy = initialize_model(cls.device, cls.env, hyperparameters)
        logger = Logger(n_envs, logdir, use_wandb=False, has_vq=False, algo=hyperparameters["algo"])
        logger.max_steps = 1000

        storage = GoalSeekerStorage(cls.obs_shape, cls.n_steps, n_envs,
                                    cls.device, continuous_actions=False,
                                    act_shape=policy.action_size)

        cls.agent = GoalSeeker(cls.env, policy, logger, storage, cls.device,
                    1, **hyperparameters)

    def test_ipl_icm(self):
        self.agent.train(int(1e5))


if __name__ == '__main__':
    unittest.main()
