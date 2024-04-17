import os
import argparse

import numpy as np
from matplotlib import pyplot as plt

if os.name != "nt":
    from pysr import PySRRegressor

from helper_local import add_symbreg_args, get_config, DictToArgs, sigmoid, entropy_from_binary_prob
from symbolic_regression import run_neurosymbolic_search, load_nn_policy, generate_data, DeterministicNeuralAgent, \
    NeuralAgent


def run_deterministic_agent():
    # logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    # logdir = "logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"
    logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"
    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, 32)

    nn_agent = NeuralAgent(policy)
    nn_score_train = test_agent(nn_agent, env, "Neural Train", 300)
    nn_score_test = test_agent(nn_agent, test_env, "Neural Test", 300)

    dn_agent = DeterministicNeuralAgent(policy)
    dn_score_train = test_agent(dn_agent, env, "DetNeural Train", 300)
    dn_score_test = test_agent(dn_agent, test_env, "DetNeural Test", 300)



def run_saved_model():
    # data_size = 1000
    # n_envs = 32
    # rounds = 300

    logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    symbdir = os.path.join(logdir, "symbreg/2024-04-12__17-38-41/")
    # symbdir = os.path.join(logdir, "symbreg/2024-04-11__15-12-07/")
    # logdir = "logs/train/cartpole/cartpole/2024-04-15__15-45-47__seed_6033"
    # symbdir = os.path.join(logdir, "symbreg/2024-04-15__16-36-44/")
    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    pysr_model = PySRRegressor.from_file(pickle_filename)

    cfg = get_config(symbdir)
    args = DictToArgs(cfg)

    policy, env, sampler, symbolic_agent_constructor, test_env, test_agent = load_nn_policy(logdir, args.n_envs)
    X, Y, V = generate_data(policy, sampler, env, int(1000), args)
    # ns_agent = symbolic_agent_constructor(pysr_model, policy, args.stochastic, None)
    # nn_agent = NeuralAgent(policy)
    # ns_score_train = test_agent(ns_agent, env, "NeuroSymb Train", args.rounds)
    # nn_score_train = test_agent(nn_agent, env, "Neural    Train", args.rounds)

    Y, Y_hat = plot_action_entropy_vs_pole_angle(X, Y, args, policy, pysr_model, sampler, symbdir)



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


def plot_action_entropy_vs_pole_angle(X, Y, args, policy, pysr_model, sampler, symbdir):
    X[:, 2] = (np.random.rand(X.shape[0]) - .5) * 28 / 360 * 2 * np.pi
    Y_hat = pysr_model.predict(X)
    _, Y, _, _ = sampler(policy, X, args.stochastic)
    p_hat = sigmoid(Y_hat)
    p = sigmoid(Y)
    ent_hat = entropy_from_binary_prob(p_hat)
    ent = entropy_from_binary_prob(p)
    pole_degrees = X[:, 2] * 360 / (2 * np.pi)
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].scatter(x=pole_degrees, y=ent)
    ax[0].set_title('Neural')
    ax[1].scatter(x=pole_degrees, y=ent_hat)
    ax[1].set_title('Symbolic')

    for _ax in ax:
        _ax.vlines([9,-9], 0, max(np.nanmax(ent),np.nanmax(ent_hat)), linestyles='dashed')

    fig.text(0.5, 0.04, 'Pole Angle Degrees', ha='center')
    fig.text(0.04, 0.5, 'Action Entropy', va='center', rotation='vertical')
    plt.savefig(os.path.join(symbdir, "pole_angle_entropy.png"))
    plt.show()
    return Y, Y_hat


def run_symb_reg_local():
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)

    args = parser.parse_args()

    args.data_size = 100
    args.iterations = 1
    # args.logdir = "logs/train/boxworld/boxworld/2024-04-08__12-29-17__seed_6033"
    # args.logdir = "logs/train/cartpole/cartpole/2024-03-28__11-49-51__seed_6033"
    args.logdir = "logs/train/coinrun/coinrun-hparams/2024-03-27__18-20-55__seed_6033"

    args.n_envs = 4
    args.rounds = 10
    args.binary_operators = ["+", "-", "*", "greater", "/"]
    args.unary_operators = ["sin", "relu", "log", "exp", "sign", "sqrt", "square"]

    args.denoise = False
    args.use_wandb = True
    args.wandb_tags = ["stochastic", "coinrun"]
    args.weight_metric = "value"
    args.wandb_name = "manual"
    # args.populations = 24
    args.model_selection = "best"
    args.ncycles_per_iteration = 2000
    args.bumper = True
    args.loss_function = "capped_sigmoid"
    args.stochastic = False
    run_neurosymbolic_search(args)


if __name__ == "__main__":
    run_saved_model()
    # run_deterministic_agent()
    # run_symb_reg_local()
