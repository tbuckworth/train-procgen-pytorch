import unittest

import numpy as np

from cartpole.cartpole_pre_vec import CartPoleVecEnv


class TestCartPoleVecEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.n_env = 160
        cls.env = CartPoleVecEnv(cls.n_env)
        cls.n_acts = cls.env.action_space.n
        cls.n_envs = cls.env.num_envs
        cls.env.reset()
    def test_step(self):
        act = 0
        actions = np.full(self.n_envs, act)
        self.env.step(actions)



if __name__ == '__main__':
    unittest.main()
