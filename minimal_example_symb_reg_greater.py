import numpy as np
import sympy
from pysr import PySRRegressor

if __name__ == "__main__":
    x = np.random.uniform(-1, 1, size=100).reshape((50, 2))
    y = x[:, 1] ** 2
    model = PySRRegressor(
        equation_file="symbreg/symbreg.csv",
        niterations=1,
        binary_operators=["greater"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        extra_sympy_mappings={"greater": lambda x, y: sympy.Piecewise((1.0, x > y), (0.0, True))}
    )
    model.fit(x, y)
    print(model)