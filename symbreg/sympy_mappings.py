import os

import sympy

from windows_dll_setup import windows_dll_setup_for_pysr

windows_dll_setup_for_pysr()
from pysr import sympy2torch, PySRRegressor
import torch
from sympy import symbols, Piecewise, GreaterThan, exp, sign, log, sin
import numpy as np
import re

from common.model import NBatchPySRTorch
# from graph_sr import test_agent_mean_reward
from helper_local import get_latest_file_matching
from symbolic_regression import load_nn_policy
from symbreg.extra_mappings import get_extra_torch_mappings


def trial_cust():
    x, y = symbols("x y")
    expression = Piecewise((1.0, x > y), (0.0, True))
    expression = GreaterThan(x, y)
    expression = exp(sign(0.44796443))*exp(sign(0.44796443))
    expression = exp(2)
    expression = sin(1)
    expression = sin(sign(-0.041662704))
    module = sympy2torch(expression, [x, y])#, extra_torch_mappings=get_extra_torch_mappings())
    X = torch.rand(100, 2).float() * 10

    torch_out = module(X)

    true_out = torch.where(X[:, 0] > X[:, 1], 1.0, 0.0)

    result = np.testing.assert_array_almost_equal(true_out.detach(), torch_out.detach(), decimal=4)


def forward_batches(self, X):
    if self._selection is not None:
        X = X[..., self._selection]
    symbols = {symbol: X[..., i] for i, symbol in enumerate(self.symbols_in)}
    return self._node(symbols)

def load_and_test():
    symbdir = get_latest_file_matching(r"\d*-\d*-\d*__", 1, folder="../logs/train/cartpole/test/2024-05-29__14-54-48__seed_6033/symbreg")
    logdir = re.search(r"(logs.*)symbreg", symbdir).group(1)
    symbdir = os.path.join(symbdir, "msg")
    symbdir = get_latest_file_matching(r"\d*-\d*-\d*__",1, symbdir)
    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    msg_model = PySRRegressor.from_file(pickle_filename, extra_torch_mappings=get_extra_torch_mappings())
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs=100)
    nn_agent = symbolic_agent_constructor(policy)
    rounds = 300
    # nn_score_train = test_agent_mean_reward(nn_agent, env, "Neural    Train", rounds, seed=0)
    # nn_score_test = test_agent_mean_reward(nn_agent, test_env, "Neural     Test", rounds, seed=0)
    return
    # x = env.reset()
    # obs = torch.FloatTensor(x).to(policy.device)
    # a = policy.actions_like(obs, 0)
    # n, x0 = policy.transition_model.prep_input(obs)
    # msg_in = policy.transition_model.vectorize_for_message_pass(a, n, x0)
    #
    # mi = msg_in.reshape((2*9*9,5))
    #
    # messages = policy.transition_model.messenger(msg_in)
    #
    # msg_py = msg_model.pytorch()
    #
    # wrapped = NBatchPySRTorch(msg_py)
    # print(wrapped(msg_in).shape)
    # ns_agent = symbolic_agent_constructor(wrapped, None, policy)

    # # policy.transition_model.messenger = msg_py
    # mpy = msg_py(mi)
    # msympy = msg_model.predict(mi)
    # mpynp = mpy.detach().numpy()
    #
    # msg_py.bark = forward_batches.__get__(msg_py, _SingleSymPyModule)
    #
    # msg_py.forward = forward_batches
    #
    # msg_py(msg_in).shape
    # forward_batches(msg_py, msg_in).shape
    # messages.shape



    policy.transition_model()

    ns_score_train = test_agent_mean_reward(ns_agent, env, "NeuroSymb Train", 10)
    ns_score_train = test_agent_mean_reward(ns_agent, test_env, "NeuroSymb Train", 10)
    nn_agent



    print("halt")

if __name__ == "__main__":
    trial_cust()
