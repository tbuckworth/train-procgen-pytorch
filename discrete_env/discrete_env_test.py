import unittest

import numpy as np

from discrete_env.acrobot_pre_vec import AcrobotVecEnv
from discrete_env.cartpole_pre_vec import CartPoleVecEnv
from discrete_env.mountain_car_pre_vec import MountainCarVecEnv, create_mountain_car
from helper_local import DictToArgs

class BaseDiscreteEnvTest(unittest.TestCase):
    def setUp(cls) -> None:
        cls.env = cls.env_cons(n_envs=2, max_steps=50, render_mode="human")
        cls.env.reset(seed=np.random.randint(0,1000))

    def run_step(self):
        for i in range(100):
            actions = [self.env.action_space.sample() for _ in range(self.env.n_envs)]
            obs, rew, done, info = self.env.step(actions)
            print(i)
            if done[0]:
                print("done")

class TestMountainCar(BaseDiscreteEnvTest):

    def setUp(cls) -> None:
        cls.env_cons = MountainCarVecEnv
        super(TestMountainCar, cls).setUp()
    def test_step(self):
        self.run_step()

class TestAcrobot(BaseDiscreteEnvTest):
    def setUp(cls) -> None:
        cls.env_cons = AcrobotVecEnv
        super(TestAcrobot, cls).setUp()
    def test_step(self):
        self.run_step()



class TestCartpole(BaseDiscreteEnvTest):
    def setUp(cls) -> None:
        cls.env_cons = CartPoleVecEnv
        super(TestCartpole, cls).setUp()
    def test_step(self):
        self.run_step()



if __name__ == '__main__':
    unittest.main()
