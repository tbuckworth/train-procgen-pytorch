"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import torch
import math
from typing import Optional

import numpy as np
from discrete_env.helper_pre_vec import StartSpace, override_value, assign_env_vars
from discrete_env.pre_vec_env import PreVecEnv, create_pre_vec
from helper_local import DictToArgs


# Analytic Solution:
# 3 * angle + angle_velocity > 0


class CartPoleVecEnv(PreVecEnv):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, n_envs,
                 degrees=12,
                 h_range=2.4,
                 min_gravity=9.8,
                 max_gravity=10.4,
                 min_pole_length=0.5,
                 max_pole_length=1.0,
                 min_cart_mass=1.0,
                 max_cart_mass=1.5,
                 min_pole_mass=0.1,
                 max_pole_mass=0.2,
                 min_force_mag=10.,
                 max_force_mag=10.,
                 max_steps=500,
                 seed=0,
                 render_mode: Optional[str] = None, ):

        n_actions = 2
        self.reward = torch.ones((n_envs))
        self.info = [{"env_reward": self.reward[i]} for i in range(len(self.reward))]
        self.min_gravity = min_gravity
        self.max_gravity = max_gravity
        self.min_cart_mass = min_cart_mass
        self.max_cart_mass = max_cart_mass
        self.min_pole_mass = min_pole_mass
        self.max_pole_mass = max_pole_mass
        self.min_pole_length = min_pole_length
        self.max_pole_length = max_pole_length
        self.min_force_mag = min_force_mag
        self.max_force_mag = max_force_mag
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.degrees = degrees
        self.h_range = h_range
        # Angle at which to fail the episode
        self.theta_threshold_radians = degrees * 2 * math.pi / 360
        self.x_threshold = h_range

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        self.high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                self.max_gravity,
                self.max_pole_length,
                self.max_cart_mass,
                self.max_pole_mass,
                self.max_force_mag,
            ],
            dtype=np.float32,
        )
        self.low = -self.high
        self.i_g = 4
        self.i_pl = 5
        self.i_mc = 6
        self.i_mp = 7
        self.i_fm = 8
        self.low[self.i_g] = self.min_gravity
        self.low[self.i_pl] = self.min_pole_length
        self.low[self.i_mc] = self.min_cart_mass
        self.low[self.i_mp] = self.min_pole_mass
        self.low[self.i_fm] = self.min_force_mag

        self.start_space = StartSpace(
            low=[-0.05, -0.05, -0.05, -0.05,
                 self.min_gravity, self.min_pole_length, self.min_cart_mass,
                 self.min_pole_mass, self.min_force_mag],
            high=[0.05, 0.05, 0.05, 0.05,
                  self.max_gravity, self.max_pole_length, self.max_cart_mass,
                  self.max_pole_mass, self.max_force_mag],
            np_random=self._np_random)

        self.customizable_params = [
            "degrees",
            "h_range",
            "min_gravity",
            "max_gravity",
            "max_steps",
            "min_cart_mass",
            "max_cart_mass",
            "min_pole_mass",
            "max_pole_mass",
            "min_pole_length",
            "max_pole_length",
            "min_force_mag",
            "max_force_mag",
            "tau",
            "kinematics_integrator",
        ]

        super().__init__(n_envs, n_actions, "CartPole", max_steps, seed, render_mode)

    def get_ob_names(self):
        return [
            "Cart Position",
            "Cart Velocity",
            "Pole Angle",
            "Pole Angular Velocity",
            "Gravity",
            "Pole Length",
            "Cart Mass",
            "Pole Mass",
            "Action Force",
        ]

    def rew_func(self, state):
        return torch.ones((state.shape[:-1]))

    def transition_model(self, action):
        x, x_dot, theta, theta_dot, gravity, pole_length, mass_cart, mass_pole, force_mag = self.state.T
        force = torch.ones((self.n_envs))
        force[action == 0] = -1
        force *= force_mag
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        polemass_length = mass_pole * pole_length
        total_mass = mass_pole + mass_cart

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + polemass_length * theta_dot ** 2 * sintheta
               ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
                pole_length * (4.0 / 3.0 - mass_pole * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = torch.vstack((x, x_dot, theta, theta_dot, gravity, pole_length, mass_cart, mass_pole, force_mag)).T
        self.terminated = self.done_func(self.state)

    def done_func(self, state):
        xt, _, thetat, _, _, _, _, _, _ = state.T
        x = xt.T
        theta = thetat.T
        oob = torch.bitwise_or(x < -self.x_threshold,
                            x > self.x_threshold)
        theta_oob = torch.bitwise_or(theta < -self.theta_threshold_radians,
                                  theta > self.theta_threshold_radians)
        return torch.bitwise_or(oob, theta_oob)

    def render_unique(self):
        if self.state is None:
            return False
        x = self.state[0]
        pole_length = x[self.i_pl]

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * pole_length)
        cartwidth = 50.0
        cartheight = 30.0
        self.surf = self.pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        self.gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        self.gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = self.pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        self.gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        self.gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))
        self.gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        self.gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        self.gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        return True

    def generate_pixels(self):
        if self.state is None:
            return False
        x = self.state
        pole_length = x[:, self.i_pl]

        world_width = self.x_threshold * 2
        scale = 64 / world_width
        polewidth = np.full((self.n_envs,), 10.0 * .16)
        polelen = scale * (2 * pole_length)
        cartwidth = np.full((self.n_envs,), 50.0 * .16)
        cartheight = np.full((self.n_envs,), 30.0 * .16)
        self.surf = np.full((self.n_envs, 64, 64, 3), 255)
        # self.surf.fill((255, 255, 255))
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[:, 0] * scale + 64 / 2.0  # MIDDLE OF CART
        carty = np.full((self.n_envs,), 100 * .16)  # TOP OF CART

        l = np.tile(l, (self.n_envs, 64, 64))
        r = np.tile(r, (1, 64, 64))
        t = np.tile(t, (1, 64, 64))
        b = np.tile(b, (1, 64, 64))

        raw_index = np.arange(64)
        col_index = np.tile(raw_index, (self.n_envs, 64, 1))
        row_index = col_index.transpose((0, 2, 1))

        row_index >= l

        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]

        self.gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        self.gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = self.pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        self.gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        self.gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))
        self.gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        self.gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        self.gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        return True

    def get_action_lookup(self):
        return {
            0: "push left",
            1: "push right",
        }


# def create_cartpole(args, hyperparameters, is_valid=False):
#     if args is None:
#         args = DictToArgs({"render": False})
#     n_envs = hyperparameters.get("n_envs", 32)
#     # The second values are for test envs, if one value, it is used for both.
#
#     # The above params will be applied, unless the hyperparameters override them.
#     env_args = assign_env_vars(hyperparameters, is_valid, param_range)
#     env_args["n_envs"] = n_envs
#     env_args["render_mode"] = "human" if args.render else None
#     return CartPoleVecEnv(**env_args)

def create_cartpole(args, hyperparameters, is_valid=False):
    param_range = {
        "degrees": [12],
        "h_range": [2.4],
        "min_gravity": [9.8, 10.4],
        "max_gravity": [10.4, 24.8],
        "min_pole_length": [0.5, 1.0],
        "max_pole_length": [1.0, 2.0],
        "min_cart_mass": [1.0, 2.],
        "max_cart_mass": [1.5, 3.],
        "min_pole_mass": [0.1, .2],
        "max_pole_mass": [0.2, .4],
        "min_force_mag": [10.],
        "max_force_mag": [10.],
    }
    return create_pre_vec(args, hyperparameters, param_range, CartPoleVecEnv, is_valid)
