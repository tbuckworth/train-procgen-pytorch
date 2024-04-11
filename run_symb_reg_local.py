import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor

from helper_local import add_symbreg_args, get_config, DictToArgs, sigmoid
from symbolic_regression import run_neurosymbolic_search, load_nn_policy, generate_data


def run_saved_model():
    # data_size = 1000
    # n_envs = 32
    # rounds = 300

    logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    symbdir = os.path.join(logdir, "symbreg/2024-04-11__15-12-07/")
    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    pysr_model = PySRRegressor.from_file(pickle_filename)

    cfg = get_config(symbdir)
    args = DictToArgs(cfg)

    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, args.n_envs)
    X, Y, V = generate_data(policy, sampler, env, int(args.data_size), args)
    ns_agent = symbolic_agent_constructor(pysr_model, policy, args.stochastic)
    # ns_score_train = test_agent(ns_agent, env, "NeuroSymb Train", args.rounds)

    Y_hat = pysr_model.predict(X)

    plt.scatter(sigmoid(Y), sigmoid(Y_hat), c=V)
    plt.scatter(sigmoid(Y), sigmoid(Y))
    plt.show()


    plt.scatter(Y, Y_hat, c=V)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend(['Value'])
    plt.show()


    p = sigmoid(Y)
    Y_act = np.int32(np.random.random(p.shape) < p)

    p_hat = sigmoid(Y_hat)
    Y_hat_act = np.int32(np.random.random(p_hat.shape) < p)

    print(f"same action:{np.sum(Y_act == Y_hat_act) / len(Y_act)}")

    plt.scatter(Y_act - Y_hat_act, V)
    plt.show()
    plt.hist(V[Y_act == Y_hat_act], bins=50)
    plt.hist(V[Y_act != Y_hat_act], bins=50)
    plt.show()

    plt.scatter(Y_hat, V, c=Y_hat_act)
    plt.show()

    plt.scatter(Y, V, c=Y_act)
    plt.show()

    print("done")


def run_symb_reg_local():
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()

    args.data_size = 100
    args.iterations = 5
    # args.logdir = "logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"
    args.logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    args.n_envs = 32
    args.rounds = 300
    args.binary_operators = ["+", "-", "*", "greater"]
    args.unary_operators = []
    args.denoise = True
    args.use_wandb = True
    args.wandb_tags = ["test"]
    args.wandb_name = "test"
    # args.populations = 24
    args.model_selection = "best"

    run_neurosymbolic_search(args)


if __name__ == "__main__":
    run_symb_reg_local()
