import torch

from common.env.env_constructor import get_env_constructor
from helper_local import get_config, get_agent_constructor, latest_model_path
from hyperparameter_optimization import run_next_hyperparameters


def main():
    better_unfinished = "logs/train/coinrun/sae/2024-12-17__15-02-35__seed_6033"
    sae_logdir = "logs/train/coinrun/sae/2024-12-17__14-53-45__seed_6033"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(sae_logdir)
    cfg["num_timesteps"] = 0
    cfg["use_wandb"] = False
    cfg["render"] = True
    cfg["n_envs"] = 2
    agent = run_next_hyperparameters(cfg)

    sae_model = latest_model_path(sae_logdir, "sae")
    checkpoint = torch.load(sae_model, map_location=device)
    agent.sae.load_state_dict(checkpoint["model_state_dict"])

    linear_model = latest_model_path(sae_logdir, "linear")
    checkpoint = torch.load(linear_model, map_location=device)
    agent.linear_model.load_state_dict(checkpoint["model_state_dict"])



    print("inspect agent")

if __name__ == "__main__":


    # create_venv = get_env_constructor(env_name)
    # env = create_venv(args, hyperparameters)
    # env_valid = create_venv(args, hyperparameters, is_valid=True)
    #
    # AGENT = get_agent_constructor("sae")
    # AGENT(env, policy, logger, storage, device,
    #               num_checkpoints,
    #               env_valid=env_valid,
    #               storage_valid=storage_valid,
    #               **hyperparameters)
    main()
