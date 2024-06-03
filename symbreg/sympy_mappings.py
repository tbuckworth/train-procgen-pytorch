import os

import sympy
from pysr import sympy2torch, PySRRegressor
import torch
from sympy import symbols, Piecewise, GreaterThan
import numpy as np
import re

from graph_sr import test_agent_mean_reward
from helper_local import get_latest_file_matching
from symbolic_regression import load_nn_policy
from symbreg.extra_mappings import get_extra_torch_mappings


def test_cust():
    x, y = symbols("x y")
    expression = Piecewise((1.0, x > y), (0.0, True))
    expression = GreaterThan(x, y)
    module = sympy2torch(expression, [x, y], extra_torch_mappings={
        # sympy.StrictGreaterThan: torch.greater,
        sympy.Piecewise: lambda x, y: torch.where(x[1], x[0], y[0]),
        # sympy.functions.elementary.piecewise.ExprCondPair: tuple,
        # sympy.logic.boolalg.BooleanTrue: torch.BoolType,
    })
    print(module)
    # >> _SingleSymPyModule(expression=Piecewise((1.0, x > y), (0.0, True)))

    X = torch.rand(100, 2).float() * 10

    torch_out = module(X)

    true_out = torch.where(X[:, 0] > X[:, 1], 1.0, 0.0)

    result = np.testing.assert_array_almost_equal(true_out.detach(), torch_out.detach(), decimal=4)


def load_and_test():
    symbdir = get_latest_file_matching(r"\d*-\d*-\d*__", 1, folder="../logs/train/cartpole/test/2024-05-29__14-54-48__seed_6033/symbreg")
    logdir = re.search(r"(logs.*)symbreg", symbdir).group(1)
    symbdir = os.path.join(symbdir, "msg")
    symbdir = get_latest_file_matching(r"\d*-\d*-\d*__",1, symbdir)
    pickle_filename = os.path.join(symbdir, "symb_reg.pkl")
    msg_model = PySRRegressor.from_file(pickle_filename, extra_torch_mappings=get_extra_torch_mappings())
    policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs=2)
    ns_agent = symbolic_agent_constructor(msg_model.pytorch(), None, policy)

    x = env.reset()
    obs = torch.FloatTensor(x).to(policy.device)
    a = policy.actions_like(obs, 0)
    n, x0 = policy.transition_model.prep_input(obs)
    msg_in = policy.transition_model.vectorize_for_message_pass(a, n, x0)
    messages = policy.transition_model.messenger(msg_in)
    msg_py = msg_model.pytorch()
    policy.transition_model.messenger = msg_py
    msg_py(msg_in).shape


    policy.transition_model()

    ns_score_train = test_agent_mean_reward(ns_agent, env, "NeuroSymb Train", 10)
    ns_score_train = test_agent_mean_reward(ns_agent, test_env, "NeuroSymb Train", 10)




    print("halt")

if __name__ == "__main__":
    load_and_test()
