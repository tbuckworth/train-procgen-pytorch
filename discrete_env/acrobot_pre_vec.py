"""classic Acrobot task"""
from typing import Optional

import numpy as np
from numpy import cos, pi, sin

import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

from discrete_env.helper_pre_vec import StartSpace, assign_env_vars
from discrete_env.pre_vec_env import PreVecEnv, create_pre_vec
from helper_local import DictToArgs


# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class AcrobotVecEnv(PreVecEnv):
    """
    ## Description

    The Acrobot environment is based on Sutton's work in
    ["Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
    and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
    The system consists of two links connected linearly to form a chain, with one end of
    the chain fixed. The joint between the two links is actuated. The goal is to apply
    torques on the actuated joint to swing the free end of the linear chain above a
    given height while starting from the initial state of hanging downwards.

    As seen in the **Gif**: two blue links connected by two green joints. The joint in
    between the two links is actuated. The goal is to swing the free end of the outer-link
    to reach the target height (black horizontal line above system) by applying torque on
    the actuator.

    ## Action Space

    The action is discrete, deterministic, and represents the torque applied on the actuated
    joint between the two links.

    | Num | Action                                | Unit         |
    |-----|---------------------------------------|--------------|
    | 0   | apply -1 torque to the actuated joint | torque (N m) |
    | 1   | apply 0 torque to the actuated joint  | torque (N m) |
    | 2   | apply 1 torque to the actuated joint  | torque (N m) |

    ## Observation Space

    The observation is a `ndarray` with shape `(6,)` that provides information about the
    two rotational joint angles as well as their angular velocities:

    | Num | Observation                  | Min                 | Max               |
    |-----|------------------------------|---------------------|-------------------|
    | 0   | Cosine of `theta1`           | -1                  | 1                 |
    | 1   | Sine of `theta1`             | -1                  | 1                 |
    | 2   | Cosine of `theta2`           | -1                  | 1                 |
    | 3   | Sine of `theta2`             | -1                  | 1                 |
    | 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
    | 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

    where
    - `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
    downwards.
    - `theta2` is ***relative to the angle of the first link.***
        An angle of 0 corresponds to having the same angle between the two links.

    The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
    A state of `[1, 0, 1, 0, ..., ...]` indicates that both links are pointing downwards.

    ## Rewards

    The goal is to have the free end reach a designated target height in as few steps as possible,
    and as such all steps that do not reach the goal incur a reward of -1.
    Achieving the target height results in termination with a reward of 0. The reward threshold is -100.

    ## Starting State

    Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
    uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.

    ## Episode End

    The episode ends if one of the following occurs:
    1. Termination: The free end reaches the target height, which is constructed as:
    `-cos(theta1) - cos(theta2 + theta1) > 1.0`
    2. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    Acrobot only has `render_mode` as a keyword for `gymnasium.make`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    # >>> import gymnasium as gym
    # >>> env = gym.make('Acrobot-v1', render_mode="rgb_array")
    # >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<AcrobotEnv<Acrobot-v1>>>>>
    # >>> env.reset(seed=123, options={"low": -0.2, "high": 0.2})  # default low=-0.1, high=0.1
    (array([ 0.997341  ,  0.07287608,  0.9841162 , -0.17752565, -0.11185605,
           -0.12625128], dtype=float32), {})

    ```

    By default, the dynamics of the acrobot follow those described in Sutton and Barto's book
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/11/node4.html).
    However, a `book_or_nips` parameter can be modified to change the pendulum dynamics to those described
    in the original [NeurIPS paper](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html).

    ```python
    # To change the dynamics as described above
    env.unwrapped.book_or_nips = 'nips'
    ```

    See the following note for details:

    > The dynamics equations were missing some terms in the NIPS paper which are present in the book.
      R. Sutton confirmed in personal correspondence that the experimental results shown in the paper and the book were
      generated with the equations shown in the book. However, there is the option to run the domain with the paper equations
      by setting `book_or_nips = 'nips'`

    ## Version History

    - v1: Maximum number of steps increased from 200 to 500. The observation space for v0 provided direct readings of
    `theta1` and `theta2` in radians, having a range of `[-pi, pi]`. The v1 observation space as described here provides the
    sine and cosine of each angle instead.
    - v0: Initial versions release

    ## References
    - Sutton, R. S. (1996). Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding.
        In D. Touretzky, M. C. Mozer, & M. Hasselmo (Eds.), Advances in Neural Information Processing Systems (Vol. 8).
        MIT Press. https://proceedings.neurips.cc/paper/1995/file/8f1d43620bc6bb580df6e80b0dc05c48-Paper.pdf
    - Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    dt = 0.2

    AVAIL_TORQUE = np.array([-1.0, 0.0, +1])

    screen_width = 500
    screen_height = 500

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None

    def __init__(self, n_envs,
                 torque_noise_max=0.0,
                 gravity=None,
                 link_length_1=None,  # [m]
                 link_length_2=None,  # [m]
                 link_mass_1=None,  #: [kg] mass of link 1
                 link_mass_2=None,  #: [kg] mass of link 2
                 link_com_pos_1=None,  #: [m] position of the center of mass of link 1
                 link_com_pos_2=None,  #: [m] position of the center of mass of link 2
                 link_moi=None,  #: moments of inertia for both links
                 max_vel_1=4 * pi,
                 max_vel_2=9 * pi,
                 max_steps=500,
                 unprocessed_features=False,
                 seed=0,
                 render_mode: Optional[str] = None):
        self.unprocessed_features = unprocessed_features
        self.gravity = gravity
        self.link_length_1 = link_length_1
        self.link_length_2 = link_length_2
        self.link_mass_1 = link_mass_1
        self.link_mass_2 = link_mass_2
        self.link_com_pos_1 = link_com_pos_1
        self.link_com_pos_2 = link_com_pos_2
        self.link_moi = link_moi,
        if link_moi is None:
            self.link_moi = [1.0, 1.0]
        if link_com_pos_2 is None:
            self.link_com_pos_2 = [0.5, 0.5]
        if link_com_pos_1 is None:
            self.link_com_pos_1 = [0.5, 0.5]
        if link_mass_2 is None:
            self.link_mass_2 = [1.0, 1.5]
        if link_mass_1 is None:
            self.link_mass_1 = [1.0, 1.5]
        if link_length_2 is None:
            self.link_length_2 = [1.0, 1.5]
        if gravity is None:
            self.gravity = [9.8, 11.4]
        if link_length_1 is None:
            self.link_length_1 = [1.0, 1.5]
        self._np_random = None
        n_actions = 3

        contextual_vars = np.array(
            [self.gravity,
             self.link_length_1,
             self.link_length_2,
             self.link_mass_1,
             self.link_mass_2,
             self.link_com_pos_1,
             self.link_com_pos_2,
             self.link_moi])
        self.i_t1 = 0
        self.i_t2 = 1
        self.i_d1 = 2
        self.i_d2 = 3
        self.i_g = 4
        self.i_ll1 = 5
        self.i_ll2 = 6
        self.i_lm1 = 7
        self.i_lm2 = 8
        self.i_lcp1 = 9
        self.i_lcp2 = 10
        self.i_moi = 11
        self.max_vel_1 = max_vel_1
        self.max_vel_2 = max_vel_2

        self.max_steps = max_steps
        self.n_envs = n_envs
        self.torque_noise_max = torque_noise_max

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2], dtype=np.float32
        )
        self.high = np.concatenate((high, contextual_vars[:, -1]))
        self.low = np.concatenate((-high, contextual_vars[:, 0]))

        base = np.array([0.1, 0.1, 0.1, 0.1])

        self.start_space = StartSpace(low=np.concatenate((-base, contextual_vars[:, 0])),
                                      high=np.concatenate((base, contextual_vars[:, -1])),
                                      np_random=self._np_random)

        self.customizable_params = ["torque_noise_max",
                                    "gravity",
                                    "link_length_1",
                                    "link_length_2",
                                    "link_mass_1",
                                    "link_mass_2",
                                    "link_com_pos_1",
                                    "link_com_pos_2",
                                    "link_moi",
                                    "max_vel_1",
                                    "max_vel_2",
                                    "max_steps",
                                    ]

        self.input_adjust = -2 if not self.unprocessed_features else 0  # override because obs != state
        super().__init__(n_envs, n_actions, "Acrobot", max_steps, seed, render_mode)

    def get_action_lookup(self):
        return {
            0: "neg torque",
            1: "no torque",
            2: "pos torque",
        }

    def transition_model(self, a):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        torque = self.AVAIL_TORQUE[a]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max, size=(self.n_envs,)
            )

        # Now, augment the state with our force action, so it can be passed to
        # _dsdt
        # exclude gravity:
        s_augmented = np.concatenate((s[:, :self.i_g], np.expand_dims(torque, 1)), axis=-1)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[:, self.i_t1] = wrap(ns[:, self.i_t1], -pi, pi)
        ns[:, self.i_t2] = wrap(ns[:, self.i_t2], -pi, pi)
        ns[:, self.i_d1] = bound(ns[:, self.i_d1], -self.max_vel_1, self.max_vel_1)
        ns[:, self.i_d2] = bound(ns[:, self.i_d2], -self.max_vel_2, self.max_vel_2)

        self.state[:, :self.i_g] = ns
        self.terminated = self._terminal()
        self.reward = np.zeros(self.n_envs)
        self.reward[np.logical_not(self.terminated)] = -1.0

        self.info = [{'env_reward': self.reward[i]} for i in range(self.n_envs)]

    def _get_ob(self):
        if self.unprocessed_features:
            return self.state
        s = self.state
        return np.concatenate((np.array(
            [
                cos(s[:, self.i_t1]),
                sin(s[:, self.i_t1]),
                cos(s[:, self.i_t2]),
                sin(s[:, self.i_t2]),
            ]), s[:, self.i_d1:].T)).T

    def _terminal(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return -cos(s[:, self.i_t1]) - cos(s[:, self.i_t2] + s[:, self.i_t1]) > 1.0

    def _dsdt(self, s_augmented):
        theta1, theta2, dtheta1, dtheta2, g, l1, l2, m1, m2, lc1, lc2, link_moi = self.state.T

        I1 = link_moi
        I2 = link_moi
        a = s_augmented[:, -1]
        d1 = (
                m1 * lc1 ** 2
                + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2))
                + I1
                + I2
        )
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
                -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
                + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                               a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2
                       ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, np.zeros_like(dtheta1)

    def render_unique(self):

        self.surf = self.pygame.Surface((self.screen_width, self.screen_width))
        self.surf.fill((255, 255, 255))
        s = self.state[0]
        link_length_1 = s[self.i_ll1]
        link_length_2 = s[self.i_ll2]

        bound = link_length_1 + link_length_2 + 0.2  # 2.2 for default
        scale = self.screen_width / (bound * 2)
        offset = self.screen_width / 2

        if s is None:
            return None

        p1 = [
            -link_length_1 * cos(s[0]) * scale,
            link_length_1 * sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - link_length_2 * cos(s[0] + s[1]) * scale,
            p1[1] + link_length_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [link_length_1 * scale, link_length_2 * scale]

        self.pygame.draw.line(
            self.surf,
            start_pos=(-2.2 * scale + offset, 1 * scale + offset),
            end_pos=(2.2 * scale + offset, 1 * scale + offset),
            color=(0, 0, 0),
        )

        for (x, y), th, llen in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = self.pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            self.gfxdraw.aapolygon(self.surf, transformed_coords, (0, 204, 204))
            self.gfxdraw.filled_polygon(self.surf, transformed_coords, (0, 204, 204))

            self.gfxdraw.aacircle(self.surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            self.gfxdraw.filled_circle(self.surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
        return True


def wrap(x, m, M):
    """Wraps `x` so m <= x <= M; but unlike `bound()` which
    truncates, `wrap()` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m

    while np.any(x > M):
        x[x > M] -= diff
    while np.any(x < m):
        x[x < M] += diff
    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    xo = x.copy()
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    xo[xo < m] = m
    xo[xo > M] = M
    return xo
    # return min(max(x, m), M)


def rk4(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

    Example for 2D system:

        # >>> def derivs(x):
        # ...     d1 =  x[0] + 2*x[1]
        # ...     d2 =  -3*x[0] + 4*x[1]
        # ...     return d1, d2
        #
        # >>> dt = 0.0005
        # >>> t = np.arange(0.0, 2.0, dt)
        # >>> y0 = (1,2)
        # >>> yout = rk4(derivs, y0, t)

    Args:
        derivs: the derivative of the system and has the signature `dy = derivs(yi)`
        y0: initial state vector
        t: sample times

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = y0.shape[-1]
    except TypeError:
        yout = np.zeros((len(y0), len(t),), np.float_)
    else:
        yout = np.zeros((len(y0), len(t), Ny), np.float_)

    yout[:, 0] = y0

    for i in np.arange(len(t) - 1):
        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[:, i]

        k1 = np.asarray(derivs(y0)).T
        k2 = np.asarray(derivs(y0 + dt2 * k1)).T
        k3 = np.asarray(derivs(y0 + dt2 * k2)).T
        k4 = np.asarray(derivs(y0 + dt * k3)).T
        yout[:, i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[:, -1][:, :4]


# def create_acrobot(args, hyperparameters, is_valid=False):
#     if args is None:
#         args = DictToArgs({"render": False, "seed": 0})
#     n_envs = hyperparameters.get('n_envs', 32)
#     param_range = {
#         "gravity": [[9.8, 10.4], [10.4, 24.8]],
#         "link_length_1": [[1., 1.5], [1.5, 2.0]],
#         "link_length_2": [[1., 1.5], [1.5, 2.0]],
#         "link_mass_1": [[1., 1.5], [1.5, 2.0]],
#         "link_mass_2": [[1., 1.5], [1.5, 2.0]],
#     }
#     env_args = assign_env_vars(hyperparameters, is_valid, param_range)
#     env_args["n_envs"] = n_envs
#     env_args["render_mode"] = "human" if args.render else None
#     env_args["seed"] = args.seed
#     return AcrobotVecEnv(**env_args)

def create_acrobot(args, hyperparameters, is_valid=False):
    param_range = {
        "gravity": [[9.8, 10.4], [10.4, 24.8]],
        "link_length_1": [[1., 1.5], [1.5, 2.0]],
        "link_length_2": [[1., 1.5], [1.5, 2.0]],
        "link_mass_1": [[1., 1.5], [1.5, 2.0]],
        "link_mass_2": [[1., 1.5], [1.5, 2.0]],
    }
    return create_pre_vec(args, hyperparameters, param_range, AcrobotVecEnv, is_valid)
