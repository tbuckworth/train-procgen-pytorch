from pysr import sympy2torch
import torch
from sympy import *
import numpy as np

if __name__ == "__main__":
    x, y = symbols("x y")
    expression = Piecewise((1.0, ))
    expression = x ** 2 + atanh(Mod(y + 1, 2) - 1) * 3.2 * z

    module = sympy2torch(expression, [x, y, z])

    print(module)
    # >> _SingleSymPyModule(expression=x**2 + 3.2*z*atanh(Mod(y + 1, 2) - 1))

    X = torch.rand(100, 3).float() * 10

    torch_out = module(X)
    true_out = X[:, 0] ** 2 + torch.atanh(torch.remainder(X[:, 1] + 1, 2) - 1) * 3.2 * X[:, 2]


    np.testing.assert_array_almost_equal(true_out.detach(), torch_out.detach(), decimal=4)
