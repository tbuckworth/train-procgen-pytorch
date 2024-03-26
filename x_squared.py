import os

import numpy as np

from common.LinearModel import XSquaredApproximator


def generate_data(low, high, shape):
    arr = np.random.uniform(low, high, size=np.prod(shape)).reshape(shape)
    return arr, arr ** 2


if __name__ == "__main__":
    low = -10.
    high = +10.
    low_t = -50.
    high_t = 50.
    n = 10000
    shape = (n, 1)
    epochs = 10000
    learning_rate = 1e-4
    n_layers = 4
    wandb_tags = ["MLP", "4 Layer", "ReLU"]

    logdir = os.path.join('logs', 'train', 'x_squared', 'x_squared')

    cfg = {"low": low,
           "high": high,
           "low_t": low_t,
           "high_t": high_t,
           "n": n,
           "dims": shape[-1],
           "learning_rate": learning_rate,
           "architecture": "MLP",
           "n_layers": n_layers,
           }
    np.save(os.path.join(logdir, "config.npy"), cfg)

    x, y = generate_data(low, high, shape)
    x_test, y_test = generate_data(low_t, high_t, shape)
    model = XSquaredApproximator(epochs, learning_rate, n_layers, logdir, cfg, wandb_tags)
    model.fit(x, y, x_test, y_test)
