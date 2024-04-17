import unittest

import torch
import yaml

from common.env.procgen_wrappers import create_env, ActionWrapper
from helper_local import initialize_model, get_combos, get_actions_from_all


def get_hyperparams(param_name):
    with open('../../hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    return hyperparameters

class ProcgenWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env_args = {"num": 2,
                    "env_name": "coinrun",
                    "num_levels": 0,
                    "start_level": 500,
                    "paint_vel_info": True,
                    "distribution_mode": "hard",
                    }
        cls.env = create_env(env_args, render=False)
        cls.device = torch.device('cpu')

    def test_action_wrapper(self):
        env2 = ActionWrapper(self.env)
        print(get_combos(env2))
        print(get_actions_from_all(env2))
        hyperparameters = get_hyperparams("hard-500-impalamha")
        model, obs_shape, policy = initialize_model(self.device, env2, hyperparameters)
        obs = env2.reset()

        for _ in range(10):
            with torch.no_grad():
                obs = torch.FloatTensor(obs).to(device=self.device)
                dist, value, hidden_state = policy(obs, None, None)
                act = dist.sample()
            obs, rew, done, info = env2.step(act)

if __name__ == '__main__':
    unittest.main()
