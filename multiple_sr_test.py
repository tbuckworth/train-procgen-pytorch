import os
import unittest

import torch

from graph_sr import get_pysr_dir, load_all_pysr
from helper_local import get_logdir_from_symbdir
from symbolic_regression import load_nn_policy


class MyTestCase(unittest.TestCase):
    def test_something(self):
        symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-08-27__19-55-01"
        logdir = get_logdir_from_symbdir(symbdir)
        policy, env, symbolic_agent_constructor, test_env = load_nn_policy(logdir, n_envs=10)

        msgdir = get_pysr_dir(symbdir, "msg")
        actdir = get_pysr_dir(symbdir, "act")
        msg_torch = load_all_pysr(msgdir, policy.device)
        act_torch = load_all_pysr(actdir, policy.device)

        ns_agent = symbolic_agent_constructor(policy, msg_torch, act_torch)

        obs = env.reset()
        obs = torch.FloatTensor(obs).to(device=policy.device)

        ns_agent.policy.graph.forward(obs)

        # x = torch.rand((10,4))
        #
        # y = msg_torch(x)
        #
        # print(y.shape)
        #
        # print("done")




if __name__ == '__main__':
    unittest.main()
