from hyperparameter_optimization import run_next_hyperparameters

# "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"

if __name__ == '__main__':
    hparams = {
        "model_file": "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033/model_200015872.pth",
        # "num_levels": 100000,
        "architecture": "sae",
        # "distribution_mode": "hard",
        "seed": 1080,
        "env_name": "coinrun",
        "exp_name": "sae",
        "param_name": "sae",
        "wandb_tags": ["sae"],
        "use_wandb": True,
        "mini_batch_size": 2048,
        "n_envs": 256,
        "sae_dim": 1024,
        "rho": 0.05,
        "sparse_coef": 1e-3,
    }
    run_next_hyperparameters(hparams)
