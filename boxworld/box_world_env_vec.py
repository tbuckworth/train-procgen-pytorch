from collections import deque

import numpy as np
from gym.spaces import Box, Discrete
from gym3.env import Env
from matplotlib import pyplot as plt
from typing_extensions import Any, List, Dict, Tuple

from boxworld.box_world_env import ACTION_LOOKUP, CHANGE_COORDINATES
from boxworld.boxworld_gen_vec import world_gen, grid_color, update_color, wall_color, goal_color, is_empty, agent_color


def colour_match(pos_colour, colour):
    return np.logical_and.reduce(pos_colour == colour, 1)


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
        self.action_names = np.array(["UP", "DOWN", "LEFT", "RIGHT"])

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

        self.reward = - self.step_cost
        self.done = self.num_env_steps == self.max_steps
        solved = np.full_like(self.done, False)
        moved = np.full_like(self.done, False)

        # Move player if the field in the moving direction is either
        pos_ind = self.position_index(new_position)
        curr_ind = self.position_index(current_position)
        left_of_ind = self.position_index(new_position - [0, 1])
        right_of_ind = self.position_index(new_position + [0, 1])

        pos_in_grid = np.bitwise_and(np.all(new_position > 0, 1),
                                     np.all(new_position <= self.n, 1))

        pos_empty = self.position_empty(pos_ind)

        pos_full_and_left_clear = np.bitwise_and(
            np.bitwise_not(pos_empty),
            np.bitwise_or(
                new_position.T[1] == 1,  # is_first_column
                self.position_empty(left_of_ind)
            )
        )
        right_colour = self.colour_at_position(right_of_ind)

        pos_is_first_key = np.bitwise_and(pos_full_and_left_clear,
                                          np.bitwise_or(
                                              colour_match(right_colour, grid_color),
                                              colour_match(right_colour, agent_color)
                                          )
                                          )

        pos_lock_status = self.get_lock_status(pos_ind)

        pos_is_lock = pos_lock_status != -1

        pos_colour = self.colour_at_position(pos_ind)

        owned_key_matches_new_pos = np.logical_and.reduce(
            np.bitwise_and(self.owned_key == pos_colour,
                           self.owned_key != grid_color), 1
        )

        unlocked_box = np.bitwise_and(pos_is_lock, owned_key_matches_new_pos)

        # try to move outside bounds
        # If not in grid then turn has been taken
        moved[np.bitwise_not(pos_in_grid)] = True

        # move to unavailable lock/key
        unavailable_key = np.bitwise_not(
            np.bitwise_or(np.bitwise_or(pos_empty, pos_is_first_key), pos_is_lock)
        )
        unavailable_lock = np.bitwise_and(pos_is_lock, np.bitwise_not(owned_key_matches_new_pos))
        unavailable = np.bitwise_or(unavailable_key, unavailable_lock)
        moved[unavailable] = True

        possible_move = np.bitwise_and(pos_in_grid, np.bitwise_not(unavailable))

        # move to empty square
        flt = np.bitwise_and(pos_empty, pos_in_grid)
        flt = np.bitwise_and(flt, np.bitwise_not(moved))
        self.player_position[flt] = new_position[flt]
        self.update_world(curr_ind, pos_ind, flt)
        moved[flt] = True

        # move to free key
        flt = np.bitwise_and(pos_is_first_key, pos_in_grid)
        flt = np.bitwise_and(flt, np.bitwise_not(moved))
        self.player_position[flt] = new_position[flt]
        self.owned_key[flt] = pos_colour[flt]
        self.world[flt, 0, 0] = pos_colour[flt]
        self.update_world(curr_ind, pos_ind, flt)
        moved[flt] = True
        self.reward[flt] += self.reward_key[flt]

        # move to lock and have correct key
        #   is distractor
        #   is on_branch
        #       is goal
        #       is not goal
        flt = np.bitwise_and(unlocked_box, np.bitwise_not(moved))
        self.player_position[flt] = new_position[flt]
        left_colour = self.colour_at_position(left_of_ind)
        is_goal = np.bitwise_and(colour_match(left_colour, goal_color), flt)
        self.reward[is_goal] += self.reward_gem[is_goal]

        is_on_branch = np.bitwise_and(pos_lock_status == 1, flt)
        self.reward[is_on_branch] += self.reward_key[is_on_branch]

        is_distractor = np.bitwise_and(pos_lock_status == 0, flt)
        self.reward[is_distractor] += self.reward_distractor[is_distractor]

        self.world[tuple(curr_ind[flt].T)] = grid_color
        self.world[tuple(left_of_ind[flt].T)] = grid_color
        self.world[tuple(pos_ind[flt].T)] = agent_color
        self.world[flt, 0, 0] = left_colour[flt]
        self.owned_key[flt] = left_colour[flt]
        moved[flt] = True

        self.done[np.bitwise_or(is_distractor, is_goal)] = True
        solved[is_goal] = True

        assert (np.all(moved))

        self.episode_reward += self.reward

        # TODO: switch dict of arrays into list of dicts?
        self.info = {
            "action.name": self.action_names[action],
            "action.moved_player": possible_move,
            "bad_transition": self.num_env_steps == self.max_steps,
            "rgb": self.world,
            "done": self.done,
            "episode": {
                "r": self.episode_reward,
                "length": self.num_env_steps,
                "solved": solved
            }
        }


        # generate new worlds:

        for i in list(np.where(self.done)[0]):
            seed = self.np_random_seed
            self.replace_world_i(i, seed)
            self.increment_seed()

        return self.world, self.reward, self.done, self.info

    def replace_world_i(self, i, seed):
        world, player_position, world_dic, num_env_steps, episode_reward, owned_key = self.generate_world(seed)
        self.world[i] = world
        self.player_position[i] = player_position
        self.world_dic[i] = world_dic
        self.num_env_steps[i] = num_env_steps
        self.episode_reward[i] = episode_reward
        self.owned_key[i] = owned_key

    def update_world(self, curr_ind, pos_ind, flt):
        self.world[tuple(curr_ind[flt].T)] = grid_color
        self.world[tuple(pos_ind[flt].T)] = agent_color

    def get_lock_status(self, pos_ind):
        return self.world_dic[tuple(pos_ind.T)]

    def position_empty(self, pos_ind):
        pos_colour = self.colour_at_position(pos_ind)
        return colour_match(pos_colour, grid_color)

    def colour_at_position(self, pos_ind):
        pos_colour = self.world[tuple(pos_ind.T)]
        return pos_colour

    def position_index(self, new_position):
        capped_position = np.minimum(np.maximum(new_position, 0), self.n + 1)
        pos_ind = np.concatenate((np.array([[i] for i in range(self.n_envs)]), capped_position), 1)
        return pos_ind

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
            world, player_position, world_dic, num_env_steps, episode_reward, owned_key = self.generate_world(
                self.np_random_seed)
            self.increment_seed()
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

    def increment_seed(self):
        self.np_random_seed += 1
        if self.n_levels > 0:
            self.np_random_seed = ((self.np_random_seed - self.start_seed) % self.n_levels) + self.start_seed

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

    ###TODO:

    def observe(self) -> Tuple[Any, Any, Any]:
        return self.reward, self.world, self.done

    def get_info(self) -> List[Dict]:
        return self.info
        # result = []
        # for env in self.envs:
        #     result.extend(env.get_info())
        # return result

    def act(self, ac: Any) -> None:
        self.step(ac)


if __name__ == "__main__":
    env = BoxWorldVec(2, 6, 2, 1, 1)
    n_acts = env.action_space.n
    n_envs = env.n_envs
    while True:
        actions = np.random.randint(0, n_acts, size=(n_envs))
        # actions[0] = 0
        env.step(actions)
        env.render()
