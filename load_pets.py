import mbrl.util.common as common_util
import torch

from common.env.env_constructor import get_pets_env_constructor
from helper_local import get_latest_file_matching, get_config, DictToArgs
from pets.pets import generate_pets_cfg_dict


def load_pets_dynamics_model(logdir):
    model_dir = get_latest_file_matching(r"model_(\d*)", 1, logdir)
    args = DictToArgs(get_config(logdir))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env_cons = get_pets_env_constructor(args.env_name)
    env = env_cons(args, {})
    env.reset(args.seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg = generate_pets_cfg_dict(args, device)
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    dynamics_model.load(model_dir)

    return dynamics_model, args, env


if __name__ == "__main__":
    logdir = "logs/pets/cartpole_continuous/2024-08-05__02-43-29__seed_6033"
    dynamics_model, args, env_cons = load_pets_dynamics_model(logdir)
