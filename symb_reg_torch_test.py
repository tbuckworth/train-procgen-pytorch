import unittest
import torch

from symb_reg_torch import create_func


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = torch.rand((100, 10))
        y = torch.rand((100, 1))
        create_func(x, y)


        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
