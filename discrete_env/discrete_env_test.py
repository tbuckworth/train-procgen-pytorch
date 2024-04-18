import unittest

from discrete_env.mountain_car_pre_vec import MountainCarVecEnv, create_mountain_car
from helper_local import DictToArgs


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # cls.env = MountainCarVecEnv(n_envs=2, render_mode="human")
        cls.env = create_mountain_car(DictToArgs({"render": True}), {}, True)
        cls.env.reset()

    def test_something(self):
        for _ in range(100):
            actions = [self.env.action_space.sample() for _ in range(self.env.n_envs)]
            obs, rew, done, info = self.env.step(actions)


if __name__ == '__main__':
    unittest.main()