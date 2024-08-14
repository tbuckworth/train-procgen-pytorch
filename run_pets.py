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
    run_pets_hyperparameters(hparams)
