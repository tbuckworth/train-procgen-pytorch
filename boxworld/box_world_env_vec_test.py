import unittest

import numpy as np
from matplotlib import pyplot as plt

from boxworld.box_world_env_vec import BoxWorldVec


class TestBoxWorldVec(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = BoxWorldVec(2, 6, 2, 1, 1, start_seed=0)
        cls.n_acts = cls.env.action_space.n
        cls.n_envs = cls.env.n_envs
        cls.action_names = cls.env.action_names
        # "UP", "DOWN", "LEFT", "RIGHT"

    def test_keys_are_locked(self):
        # left, left, down, down
        self.impossible_move([2, 2, 1, 1], render=False)

    def test_locks_dont_open_when_no_key(self):
        self.impossible_move([1], render=False)

    def test_north_boundary(self):
        self.impossible_move([0, 0, 0], render=False)

    def test_west_boundary(self):
        self.impossible_move([3, 3, 3], render=False)

    def test_south_boundary(self):
        self.impossible_move([3, 1, 1, 1, 1], render=False)

    def test_east_boundary(self):
        self.impossible_move([0, 2, 2, 2, 2], render=False)

    def test_gem_inaccessible(self):
        self.impossible_move([2, 1], render=False)

    def test_free_key_accessible(self):
        act_seq = [2, 2, 2]
        world, reward, done, info, world2, reward2, done2, info2 = self.run_sequence(act_seq, False)
        self.assertTrue(reward2[0] == 1)

    def test_distractor_ends_game(self):
        act_seq = [2, 2, 2, 1, 1, 1, 3, 3, 0]
        world, reward, done, info, world2, reward2, done2, info2 = self.run_sequence(act_seq, False)
        self.assertTrue(reward2[0] == -1)

    def test_goal_reachable(self):
        pass

    def impossible_move(self, act_seq, render):
        world, reward, done, info, world2, reward2, done2, info2 = self.run_sequence(act_seq, render)
        self.assertTrue(np.all(world[0] == world2[0]))

    def run_sequence(self, act_seq, render):
        world = self.env.world
        if render:
            self.env.render()
        for act in act_seq[:-1]:
            actions = np.full(self.n_envs, act)
            world, reward, done, info = self.env.step(actions)
        if render:
            self.env.render()
        action = np.full(self.n_envs, act_seq[-1])
        world2, reward2, done2, info2 = self.env.step(action)
        if render:
            self.env.render()
        return world, reward, done, info, world2, reward2, done2, info2


if __name__ == '__main__':
    unittest.main()
