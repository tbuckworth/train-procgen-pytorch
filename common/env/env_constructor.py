from boxworld.create_box_world import create_bw_env
from common.env.procgen_wrappers import create_procgen_env
from discrete_env.acrobot_pre_vec import create_acrobot
from discrete_env.cartpole_pre_vec import create_cartpole
from discrete_env.mountain_car_pre_vec import create_mountain_car


def get_env_constructor(env_name):
    create_venv = create_procgen_env
    if env_name == "boxworld":
        create_venv = create_bw_env
    if env_name == "cartpole":
        create_venv = create_cartpole
    if env_name == "mountain_car":
        create_venv = create_mountain_car
    if env_name == "acrobot":
        create_venv = create_acrobot
    if env_name == "lunar_lander":
        raise NotImplementedError
    return create_venv
