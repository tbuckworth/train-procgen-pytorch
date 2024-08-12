import mbrl.util.common as common_util
import torch
from mbrl import models

from common.env.env_constructor import get_pets_env_constructor
from helper_local import get_latest_file_matching, get_config, DictToArgs
from pets.pets import generate_pets_cfg_dict, load_pets_agent
from symbreg.agents import PetsSymbolicAgent
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns

def load_pets_dynamics_model(logdir):
    model_dir = get_latest_file_matching(r"model_(\d*)", 1, logdir)
    args = DictToArgs(get_config(logdir))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env_cons = get_pets_env_constructor(args.env_name)
    env = env_cons(args, {})
    env.reset(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg = generate_pets_cfg_dict(args, device)
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    dynamics_model.load(model_dir)

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.cartpole
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.cartpole
    try:
        if args.use_custom_reward_fn:
            reward_fn = env.rew_func
            term_fn = env.done_func
    except Exception as e:
        pass

    model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)
    agent = load_pets_agent(args, device, model_env)

    # agent.set_trajectory_eval_fn()

    return agent, model_env, args, env

def trajectory_eval_fn(initial_state, action_sequences):
    return model_env.evaluate_action_sequences(
        action_sequences, initial_state=initial_state, num_particles=args.num_particles
    )

if __name__ == "__main__":
    logdir = "logs/pets/cartpole_continuous/2024-08-05__02-43-29__seed_6033"
    agent, model_env, args, env = load_pets_dynamics_model(logdir)


    symb_agent = PetsSymbolicAgent(agent, model_env, args.num_particles)
    # generate env data
    obs, _ = env.reset(args.seed)
    # generate training data
    symb_agent.sample(obs)

    # do symbolic regression



