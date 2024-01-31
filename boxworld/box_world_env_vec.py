from collections import deque

import numpy as np
from gym.spaces import Box, Discrete
from gym3.env import Env
from matplotlib import pyplot as plt
from typing_extensions import Any, List, Dict, Tuple

from boxworld.box_world_env import ACTION_LOOKUP, CHANGE_COORDINATES
from boxworld.boxworld_gen_vec import world_gen, grid_color, update_color, wall_color, goal_color, is_empty


class BoxWorldVec(Env):
    def __init__(self, n_envs, n, goal_length, num_distractor, distractor_length, max_steps=10 ** 6, collect_key=True,
                 world=None, start_seed=0, n_levels=0):
        self.observation_space = Box(low=0, high=255, shape=(n + 2, n + 2, 3), dtype=np.uint8)
        self.action_space = Discrete(len(ACTION_LOOKUP))
        super().__init__(self.observation_space, self.action_space, n_envs)
        self.n_envs = n_envs
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.n = n
        self.num_pairs = goal_length - 1 + distractor_length * num_distractor
        self.collect_key = collect_key  # if True, keys are collected immediately when available

        # Penalties and Rewards
        self.step_cost = np.full(self.n_envs, 0)
        self.reward_gem = np.full(self.n_envs, 10)
        self.reward_key = np.full(self.n_envs, 1)
        self.reward_distractor = np.full(self.n_envs, -1)

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps

        # Game initialization

        self.start_seed = start_seed
        self.np_random_seed = start_seed
        self.n_levels = n_levels
        self.reset()

        self.last_frames = deque(maxlen=3)
        self.move_coordinates = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


    def seed(self, seed=None):
        self.np_random_seed = seed
        return [seed]

    def save(self):
        np.save('box_world.npy', self.world)

    def step(self, action):

        change = self.move_coordinates[action]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        self.num_env_steps += 1

        reward = - self.step_cost
        done = self.num_env_steps == self.max_steps
        solved = np.full_like(done, False)

        # Move player if the field in the moving direction is either

        possible_move = np.bitwise_and(np.any(new_position > 0, 1),
                                       np.any(new_position <= self.n, 1))

        pos_ind = np.concatenate((np.array([[i] for i in range(self.n_envs)]), new_position), 1)

        self.world[tuple(pos_ind.T)]

        if np.any(new_position < 1) or np.any(new_position >= self.n + 1):
            possible_move = False

        elif is_empty(self.world[new_position[0], new_position[1]]):
            # No key, no lock
            possible_move = True

        elif new_position[1] == 1 or is_empty(self.world[new_position[0], new_position[1] - 1]):
            # It is a key
            if is_empty(self.world[new_position[0], new_position[1] + 1]):
                # Key is not locked
                possible_move = True
                self.owned_key = self.world[new_position[0], new_position[1]].copy()
                self.world[0, 0] = self.owned_key
                if np.array_equal(self.world[new_position[0], new_position[1]], goal_color):
                    # Goal reached
                    self.world[0, 0] = wall_color
                    reward += self.reward_gem
                    solved = True
                    done = True

                else:
                    reward += self.reward_key
            else:
                possible_move = False
        else:
            # It is a lock
            if np.array_equal(self.world[new_position[0], new_position[1]], self.owned_key):
                # The lock matches the key
                possible_move = True

                if self.collect_key:
                    # goal reached
                    if np.array_equal(self.world[new_position[0], new_position[1] - 1], goal_color):
                        # Goal reached
                        self.world[new_position[0], new_position[1] - 1] = [220, 220, 220]
                        self.world[0, 0] = wall_color
                        reward += self.reward_gem
                        solved = True
                        done = True

                    else:
                        # loose old key and collect new one
                        self.owned_key = np.copy(self.world[new_position[0], new_position[1] - 1])
                        self.world[new_position[0], new_position[1] - 1] = [220, 220, 220]
                        self.world[0, 0] = self.owned_key
                        if self.world_dic[tuple(new_position)] == 0:
                            reward += self.reward_distractor
                            done = True
                        else:
                            reward += self.reward_key
                else:
                    self.owned_key = [220, 220, 220]
                    self.world[0, 0] = [0, 0, 0]
                    if self.world_dic[tuple(new_position)] == 0:
                        reward += self.reward_distractor
                        done = True
            else:
                possible_move = False
                # print("lock color is {}, but owned key is {}".format(
                #     self.world[new_position[0], new_position[1]], self.owned_key))

        if possible_move:
            self.player_position = new_position
            update_color(self.world, previous_agent_loc=current_position, new_agent_loc=new_position)

        self.episode_reward += reward

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": possible_move,
            "bad_transition": self.max_steps == self.num_env_steps,
        }
        if done:
            info["episode"] = {"r": self.episode_reward,
                               "length": self.num_env_steps,
                               "solved": solved}
        self.last_frames.append(self.world)
        obs = (self.world - grid_color[0]) / 255 * 2
        info["rgb"] = self.world.astype(np.uint8)
        return obs, reward, done, info

    def generate_world(self, seed):
        world, player_position, world_dic = world_gen(n=self.n, goal_length=self.goal_length,
                                                      num_distractor=self.num_distractor,
                                                      distractor_length=self.distractor_length,
                                                      seed=seed)
        num_env_steps = np.array([0])
        episode_reward = np.array([0])
        owned_key = np.array([[220, 220, 220]])
        return world, player_position, world_dic, num_env_steps, episode_reward, owned_key

    def reset(self):
        for i in range(self.n_envs):
            world, player_position, world_dic, num_env_steps, episode_reward, owned_key = self.generate_world(self.np_random_seed)
            self.np_random_seed += 1
            if self.n_levels > 0:
                self.np_random_seed = ((self.np_random_seed - self.start_seed) % self.n_levels) + self.start_seed
            if i == 0:
                self.world = world
                self.player_position = player_position
                self.world_dic = world_dic
                self.num_env_steps = num_env_steps
                self.episode_reward = episode_reward
                self.owned_key = owned_key
            else:
                self.world = np.concatenate((self.world, world))
                self.player_position = np.concatenate((self.player_position, player_position))
                self.world_dic = np.concatenate((self.world_dic, world_dic))
                self.num_env_steps = np.concatenate((self.num_env_steps, num_env_steps))
                self.episode_reward = np.concatenate((self.episode_reward, episode_reward))
                self.owned_key = np.concatenate((self.owned_key, owned_key))

        return (self.world - grid_color[0]) / 255 * 2

    def render(self, mode="human"):
        img = self.world[0].astype(np.uint8)
        if mode == "rgb_array":
            return img

        else:
            # from gym.envs.classic_control import rendering
            # if self.viewer is None:
            #     self.viewer = rendering.SimpleImageViewer()
            # self.viewer.imshow(img)
            # return self.viewer.isopen
            plt.imshow(img, vmin=0, vmax=255, interpolation='none')
            plt.show()

    def observe(self) -> Tuple[Any, Any, Any]:
        rews, obs, firsts = zip(*[env.observe() for env in self.envs])
        return rews, obs, firsts

    def get_info(self) -> List[Dict]:
        result = []
        for env in self.envs:
            result.extend(env.get_info())
        return result

    def act(self, ac: Any) -> None:
        self.step(ac)


if __name__ == "__main__":
    env = BoxWorldVec(2, 6, 2, 1, 1)
    n_acts = env.action_space.n
    n_envs = env.n_envs
    actions = np.random.randint(0, n_acts, size=(n_envs))
    env.step(actions)