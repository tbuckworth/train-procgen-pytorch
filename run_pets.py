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
        "hid_size": 122,
        "learning_rate": 0.000672,
        "model_batch_size": 26,
        "num_layers": 6,
        "seed": 6033,
        "use_wandb": True,
        "wandb_tags": ["pgt1"],
        "use_custom_reward_fn": False,
    }
    run_pets_hyperparameters(hparams)
