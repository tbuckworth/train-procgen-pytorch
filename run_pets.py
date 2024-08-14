import argparse

from helper_local import add_pets_args
from hyperparameter_optimization import run_pets_hyperparameters
from pets.pets import run_pets


def normal():
    parser = argparse.ArgumentParser()
    parser = add_pets_args(parser)
    args = parser.parse_args()
    run_pets(args)


if __name__ == '__main__':
    hparams = {
        "env_name": "cartpole_continuous",
        "trial_length": 500,
        "num_trials": 40,
        "hid_size": 512,
        "learning_rate": 0.000672,
        "model_batch_size": 16,
        "num_layers": 6,
        "seed": 6033,
        "use_wandb": True,
        "wandb_tags": ["pgt2"],
        "use_custom_reward_fn": False,
        "num_epochs": 1000,
        "overfit": True,
        "drop_same": True,
        "min_cart_mass": 1.0,
        "max_cart_mass": 1.0,
        "min_pole_mass": 0.1,
        "max_pole_mass": 0.1,
        "min_force_mag": 10.,
        "max_force_mag": 10.,
    }

    hparams = {
        "alpha": 0.1,
        "detect_nan": False,
        "deterministic": False,
        "dyn_model": "pets.pets_models.GraphTransitionPets",
        "elite_ratio": 0.1,
        "ensemble_size": 5,
        "env_name": "cartpole_continuous",
        "exp_name": "",
        "hid_size": 106,
        "learning_rate": 0.000688,
        "logdir": "logs/pets/cartpole_continuous/2024-08-05__02-43-29__seed_6033",
        "model_batch_size": 19,
        "num_checkpoints": 4,
        "num_epochs": 50,
        "num_iterations": 5,
        "num_layers": 5,
        "num_particles": 20,
        "num_trials": 20,
        "patience": 50,
        "planning_horizon": 15,
        "population_size": 500,
        "render": False,
        "replan_freq": 1,
        "seed": 6033,
        "trial_length": 500,
        "use_wandb": True,
        "validation_ratio": 0.05,
        "wandb_tags": ["pgt1"],
        "1": "graph-transition",
        "weight_decay": 0.00005,
    }

    run_pets_hyperparameters(hparams)
