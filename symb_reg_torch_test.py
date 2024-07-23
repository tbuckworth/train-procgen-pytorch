import unittest
import torch
from matplotlib import pyplot as plt

from symb_reg_torch import create_func, run_tree


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = torch.rand((100, 10))
        y = torch.cos(x[:, 1])**2 + torch.sin(x[:, 5])**3
        y = x[..., 3] * x[..., 5]
        model = run_tree(x, y)
        y_hat = model.forward(x)

        plt.scatter(y, y_hat)
        plt.show()

        print("done")


if __name__ == '__main__':
    unittest.main()
