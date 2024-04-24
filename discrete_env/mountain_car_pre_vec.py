"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math
from typing import Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from discrete_env.helper_pre_vec import StartSpace
from discrete_env.pre_vec_env import PreVecEnv
from helper_local import DictToArgs


class MountainCarVecEnv(PreVecEnv):
    """
    ## Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
    This version is the one with discrete actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```

    ## Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min   | Max  | Unit         |
    |-----|--------------------------------------|-------|------|--------------|
    | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
    | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |

    ## Action Space

    There are 3 discrete deterministic actions:

    - 0: Accelerate to the left
    - 1: Don't accelerate
    - 2: Accelerate to the right

    ## Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
    upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
    velocity is clipped to the range `[-0.07, 0.07]`.

    ## Reward:

    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
    penalised with a reward of -1 for each timestep.

    ## Starting State

    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
    The starting velocity of the car is always assigned to 0.

    ## Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 200.

    ## Arguments

    Mountain Car has two parameters for `gymnasium.make` with `render_mode` and `goal_velocity`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    # >>> import gymnasium as gym
    # >>> env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)  # default goal_velocity=0
    # >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<MountainCarEnv<MountainCar-v0>>>>>
    # >>> env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})  # default x_init=np.pi, y_init=1.0
    (array([-0.46352962,  0.        ], dtype=float32), {})

    ```

    ## Version History

    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self,
                 n_envs=2,
                 goal_velocity=0,
                 min_position=-1.2,
                 max_position=0.6,
                 min_start_position=-0.6,
                 max_start_position=-0.4,
                 max_speed=0.07,
                 goal_position=0.5,
                 force=0.001,
                 max_steps=500,
                 max_gravity=0.01,
                 min_gravity=0.0025,
                 render_mode: Optional[str] = None,
                 ):
        n_actions = 3
        self._np_random = None
        self.max_gravity = max_gravity
        self.min_gravity = min_gravity
        self.min_position = min_position
        self.max_position = max_position
        self.min_start_position = min_start_position
        self.max_start_position = max_start_position
        assert self.min_start_position >= self.min_position, \
            f"min_start_position ({self.min_start_position}) must be >= min_position ({self.min_position})"
        assert self.max_start_position <= self.max_position, \
            f"max_start_position ({self.max_start_position}) must be <= max_position ({self.max_position})"

        self.max_speed = max_speed
        self.goal_position = goal_position
        self.goal_velocity = goal_velocity

        self.force = force

        self.low = np.array([self.min_position, -self.max_speed, self.min_gravity], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed, self.max_gravity], dtype=np.float32)

        self.start_space = StartSpace(low=[self.min_start_position, 0, self.min_gravity],
                                      high=[self.max_start_position, 0, self.max_gravity],
                                      np_random=self._np_random)

        self.reward = np.full(n_envs, -1.0)
        self.info = [{"env_reward": self.reward[i]} for i in range(n_envs)]

        super().__init__(n_envs, n_actions, "MountainCar", max_steps, render_mode)

    def transition_model(self, action: np.array):
        position, velocity, gravity = self.state.T
        velocity += (action - 1) * self.force + np.cos(3 * position) * (-gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        velocity[np.bitwise_and(position == self.min_position, velocity < 0)] = 0

        self.terminated = np.bitwise_and(position >= self.goal_position, velocity >= self.goal_velocity)

        self.state = np.vstack((position, velocity, gravity)).T


    def get_action_lookup(self):
        return {
            0: 'acc left',
            1: 'none',
            2: 'acc right',
        }

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render_unique(self):
        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = self.pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # This is where we pick the first environment:
        pos = self.state[0, 0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        self.pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = self.pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        self.gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        self.gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = self.pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            self.gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            self.gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        self.gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        self.gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        self.gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        return True


def create_mountain_car(args, hyperparameters, is_valid=False):
    if args is None:
        args = DictToArgs({"render": False})
    n_envs = hyperparameters.get('n_envs', 32)
    env_args = {"goal_velocity": hyperparameters.get("goal_velocity", 0),
                "min_position": hyperparameters.get("min_position", -1.2),
                "max_position": hyperparameters.get("max_position", 0.6),
                "min_start_position": hyperparameters.get("min_start_position", -0.6),
                "max_start_position": hyperparameters.get("max_start_position", -0.4),
                "max_speed": hyperparameters.get("max_speed", 0.07),
                "goal_position": hyperparameters.get("goal_position", 0.5),
                "force": hyperparameters.get("force", 0.001),
                "gravity": hyperparameters.get("gravity", 0.0025),
                }
    if is_valid:
        env_args = {"goal_velocity": hyperparameters.get("goal_velocity_v", 0),
                    "min_position": hyperparameters.get("min_position_v", -1.2),
                    "max_position": hyperparameters.get("max_position_v", 0.6),
                    "min_start_position": hyperparameters.get("min_start_position_v", -0.6),
                    "max_start_position": hyperparameters.get("max_start_position_v", -0.4),
                    "max_speed": hyperparameters.get("max_speed_v", 0.07),
                    "goal_position": hyperparameters.get("goal_position_v", 0.5),
                    "force": hyperparameters.get("force_v", 0.001),
                    "gravity": hyperparameters.get("gravity_v", 0.01),
                    }
    env_args["n_envs"] = n_envs
    env_args["render_mode"] = "human" if args.render else None
    return MountainCarVecEnv(**env_args)
