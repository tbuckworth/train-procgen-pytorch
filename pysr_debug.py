import argparse
import os
from time import strftime

import numpy as np
from pysr import PySRRegressor
import torch

from common.model import NBatchPySRTorch
from double_graph_sr import find_model
from helper_local import add_symbreg_args


def run():
    x = np.random.randn(100,1)
    y = np.repeat(np.exp(-1),100)

    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)
    args = parser.parse_args()

    args.binary_operators = ["+", "-", "greater", "*", "/"]
    args.unary_operators = ["exp", "sign"]
    args.iterations = 1

    symbdir = "logs/test/" + strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(symbdir):
        os.makedirs(symbdir)

    model, _ = find_model(x, y, symbdir, "symb_reg.csv", None, args)


    msg_torch = NBatchPySRTorch(model.pytorch())

    msg_torch.forward(x)


    print("done")

if __name__ == '__main__':
    run()
