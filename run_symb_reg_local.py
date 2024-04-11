import argparse

from helper_local import add_symbreg_args
from symbolic_regression import run_neurosymbolic_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()

    args.data_size = 1000
    args.iterations = 1
    args.logdir = "logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"
    # args.logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    args.n_envs = 32
    args.rounds = 300
    args.binary_operators = ["+", "-", "greater"]
    args.unary_operators = []
    args.denoise = True
    args.use_wandb = False
    args.wandb_tags = ["test"]
    args.wandb_name = "test"
    # args.populations = 24

    run_neurosymbolic_search(args)