import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from common.LinearModel import XSquaredApproximator
from helper_local import get_config, latest_model_path
from symbolic_regression import find_model
from x_squared import generate_data


def load_model(logdir):
    device = torch.device('cpu')
    last_model = latest_model_path(logdir)
    cfg = get_config(logdir)
    xsa = XSquaredApproximator(epochs=0,
                                 learning_rate=0,
                                 depth=cfg["n_layers"],
                                 logdir=logdir,
                                 cfg=cfg,
                                 wandb_tags=None)
    xsa.model.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])
    return xsa, cfg


def generate_data_from_oracle(xsa, low, high, shape):
    arr = np.random.uniform(low, high, size=np.prod(shape)).reshape(shape)
    return arr, xsa.forward_np(arr)


def plot_x_squared(x_test, y_test, y_symb, y_nn, cfg, file_out):
    plt.axvspan(cfg["low"], cfg["high"],color='blue', alpha=0.3, label="Training Region")
    plt.scatter(x_test, y_test, color='grey', label=f"$y=x^2$", s=16)
    plt.scatter(x_test, y_symb, color='black', label="Symbolic Regression",s=4)
    plt.scatter(x_test, y_nn, color='red', label="Neural Network",s=.25)
    plt.legend()
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.title("Symbolic Regression of MLP vs MLP on $y = x^2$")
    plt.savefig(file_out, transparent=False, facecolor='white')
    plt.close()

if __name__ == "__main__":
    iterations = 40
    data_size = 10000
    save_file = "symb_reg.csv"
    logdir = "logs/train/x_squared/x_squared/2024-03-26__14-35-13__seed_2063"
    image_file = os.path.join(logdir, "x_squared.png")
    xsa, cfg = load_model(logdir)
    low = cfg["low"]
    high = cfg["high"]
    low_t = cfg["low_t"]
    high_t = cfg["high_t"]

    shape = (data_size, cfg["dims"])
    x, y = generate_data_from_oracle(xsa, low, high, shape)

    symb_model = find_model(x, y, logdir, iterations, save_file)

    x_test, y_test = generate_data(low_t, high_t, shape)
    y_symb = symb_model.predict(x_test)
    y_nn = xsa.forward_np(x_test)

    plot_x_squared(x_test, y_test, y_symb, y_nn, cfg, image_file)

