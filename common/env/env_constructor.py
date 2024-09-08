import re

from boxworld.create_box_world import create_bw_env
from common.env.mujoco_wrappers import create_humanoid, create_mujoco
from common.env.procgen_wrappers import create_procgen_env
from discrete_env.acrobot_pre_vec import create_acrobot
from discrete_env.cartpole_pre_vec import create_cartpole, create_cartpole_continuous
from discrete_env.cartpole_swing_pre_vec import create_cartpole_swing
from discrete_env.mountain_car_pre_vec import create_mountain_car
from discrete_env.pre_vec_wrappers import DeVecEnvWrapper, PetsWrapper


def get_env_constructor(env_name):
    create_venv = create_procgen_env
    if env_name == "boxworld":
        create_venv = create_bw_env
    elif env_name == "cartpole":
        create_venv = create_cartpole
    elif env_name == "cartpole_continuous":
        create_venv = create_cartpole_continuous
    elif env_name == "cartpole_swing":
        create_venv = create_cartpole_swing
    elif env_name == "mountain_car":
        create_venv = create_mountain_car
    elif env_name == "acrobot":
        create_venv = create_acrobot
    elif env_name == "lunar_lander":
        raise NotImplementedError
    elif re.search("-v", env_name):
        create_venv = create_mujoco(env_name)
    return create_venv


def get_pets_env_constructor(env_name):
    env_cons = get_env_constructor(env_name)

    def pets_cons(*args, **kwargs):
        return PetsWrapper(DeVecEnvWrapper(env_cons(*args, **kwargs)))

    return pets_cons
