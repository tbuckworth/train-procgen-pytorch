import os
import argparse

from pysr import PySRRegressor

from helper_local import add_symbreg_args
from symbolic_regression import run_neurosymbolic_search, load_nn_policy, generate_data


def run_saved_model():
    data_size = 1000
    n_envs = 32
    rounds = 300

    args = from_config

    logdir = "logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"
    pickle_filename = os.path.join(logdir, "symbreg/2024-04-10__16-03-24/symb_reg.pkl")
    pysr_model = PySRRegressor.from_file(pickle_filename)

    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, n_envs)
    X, Y, V = generate_data(policy, sampler, env, n=int(data_size), args)
    ns_agent = symbolic_agent_constructor(pysr_model, policy)
    ns_score_train = test_agent(ns_agent, env, "NeuroSymb Train", rounds)


def run_symb_reg_local():
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


if __name__ == "__main__":
    run_symb_reg_local()
