from hyperparameter_optimization import run_next_hyperparameters

# "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"

if __name__ == '__main__':
    hparams = {
        "model_file": "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033/model_200015872.pth",
        "device": "gpu",
        "num_timesteps": int(5e6),
        "num_levels": 100000,
        "distribution_mode": "hard",
        "seed": 6033,
        "epoch": 5,
        "env_name": "coinrun",
        "exp_name": "sae",
        "param_name": "sae",
        "wandb_tags": ["sae"],
        "use_wandb": True,
        "mini_batch_size": 1024,
        "n_envs": 256,
        "sae_dim": 4096,
        "rho": 0.05,
        "sparse_coef": 1e-3,
        "reduce_duplicate_actions": False,
    }
    run_next_hyperparameters(hparams)
