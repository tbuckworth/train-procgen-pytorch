import numpy as np
from pysr import PySRRegressor

if __name__ == "__main__":
    x = np.random.uniform(-1, 1, size=100).reshape((50, 2))
    y = x[:, 1] ** 2
    model = PySRRegressor(
        equation_file="symbreg/symbreg.csv",
        niterations=1,
        binary_operators=["greater"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
    model.fit(x, y)
