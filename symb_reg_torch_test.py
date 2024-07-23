import unittest
import torch
from matplotlib import pyplot as plt

from symb_reg_torch import create_func, run_tree


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = torch.rand((100, 5))
        # y = torch.cos(x[:, 1])**2 + torch.sin(x[:, 5])**3
        y = x[..., 3] * x[..., 4]
        tree = run_tree(x, y, 200, 50)
        model = tree.get_best()
        y_hat = model.forward(x)

        plt.scatter(y, y_hat)
        plt.show()

        print("done")
        [print(x.get_name()) for x in tree.all_vars]

if __name__ == '__main__':
    unittest.main()
