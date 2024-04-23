import unittest

from discrete_env.acrobot_pre_vec import AcrobotVecEnv
from discrete_env.mountain_car_pre_vec import MountainCarVecEnv, create_mountain_car
from helper_local import DictToArgs


class TestMountainCar(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = MountainCarVecEnv(n_envs=2, max_steps=100, render_mode="human")
        # cls.env = create_mountain_car(DictToArgs({"render": True, "max_steps": 20}), {}, True)
        cls.env.reset()

    def test_step(self):
        for i in range(500):
            actions = [self.env.action_space.sample() for _ in range(self.env.n_envs)]
            obs, rew, done, info = self.env.step(actions)
            print(i)
            if done[0]:
                print("done")

class TestAcrobot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = AcrobotVecEnv(n_envs=2, max_steps=50, render_mode="human")
        cls.env.reset()

    def test_step(self):
        for i in range(200):
            actions = [self.env.action_space.sample() for _ in range(self.env.n_envs)]
            obs, rew, done, info = self.env.step(actions)
            print(i)
            if done[0]:
                print("done")

if __name__ == '__main__':
    unittest.main()
