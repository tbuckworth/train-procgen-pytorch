from torch import cuda

from helper_local import DictToArgs
from symbolic_regression import run_neurosymbolic_search
from train import train_ppo

if __name__ == "__main__":

    args = DictToArgs({
        "exp_name": "mountain_car",
        "env_name": "mountain_car",
        "param_name": "mountain_car",
        "num_timesteps": int(30e6),
        "num_checkpoints": 1,
        "seed": 6033,
        "device": "gpu" if cuda.is_available() else "cpu",
        "n_envs": 1024,
        "wandb_tags": ["test", "mountain_car"],

        "render": False,
        "detect_nan": False,
        "use_valid_env": True,
        "normalize_rew": False,
        "paint_vel_info": True,
        "reduce_duplicate_actions": True,
        "use_wandb": True,
        "real_procgen": True,
        "mirror_env": False,
        
        
        "val_env_name": "mountain_car",
        "start_level": int(0),
        "num_levels": int(500),
        "distribution_mode": "easy",
        "gpu_device": int(0),
        "log_level": int(40),
        'model_file': None,
        "mut_info_alpha": None,
        "gamma": None,
        "learning_rate": None,
        "entropy_coef": None,
        "n_steps": None,
        "n_minibatch": None,
        "mini_batch_size": None,
        "wandb_name": None,
        "wandb_group": None,
        "levels": None,
        "sparsity_coef": 0.,
        "random_percent": 0,
        "key_penalty": 0,
        "step_penalty": 0,
        "rand_region": 0,
        "num_threads": 8,
    })
    logdir = train_ppo(args)

    sargs = DictToArgs({
        "logdir": logdir,
        "data_size": 100,
        "iterations": 5,

        "n_envs": 32,
        "rounds": 300,
        "binary_operators": ["+", "-", "*", "greater", "/"],
        "unary_operators": ["sin", "relu", "log", "exp", "sign", "sqrt", "square"],

        "denoise": False,
        "use_wandb": True,
        "wandb_tags": ["stochastic", "mountain_car", "test"],
        "weight_metric": "value",
        "wandb_name": "test_mountain_car",
        "wandb_group": None,
        "timeout_in_seconds": 3600,
        "populations": 24,
        "procs": 8,
        "model_selection": "best",
        "ncycles_per_iteration": 3000,
        "bumper": True,
        "loss_function": "capped_sigmoid",
        "stochastic": True,
    })

    run_neurosymbolic_search(sargs)
