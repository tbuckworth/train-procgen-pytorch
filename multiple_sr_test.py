import os
import unittest

import torch

from common.model import NBatchPySRTorchMult
from graph_sr import get_pysr_dir, pysr_from_file
from symbreg.extra_mappings import get_extra_torch_mappings


class MyTestCase(unittest.TestCase):
    def test_something(self):
        symbdir = "logs/train/cartpole/pure-graph/2024-08-23__15-44-40__seed_6033/symbreg/2024-08-27__19-55-01"
        msgdir = get_pysr_dir(symbdir, "msg")

        pickle_filename = os.path.join(msgdir, "symb_reg.pkl")
        # msg_model = PySRRegressor.from_file(pickle_filename, extra_torch_mappings=get_extra_torch_mappings())
        msg_model = pysr_from_file(pickle_filename, extra_torch_mappings=get_extra_torch_mappings())

        idx = [i for i in range(len(msg_model.equations_))]
        msg_torch = NBatchPySRTorchMult(msg_model.pytorch(idx).tolist(), cat_dim=0)

        x = torch.rand((10,4))

        y = msg_torch(x)

        print(y.shape)

        print("done")

if __name__ == '__main__':
    unittest.main()
