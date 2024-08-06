import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from common import xavier_uniform_init
from symb_reg_torch import create_func, run_tree, BinaryNode, BaseNode, ScalarNode, UnaryNode, FunctionTree

from sklearn import linear_model

class MyTestCase(unittest.TestCase):
    def test_something(self):
        # lmbda = 0.1
        x = torch.rand((100, 5))
        y = torch.cos(x[:, 1]) ** 2 + torch.sin(x[:, 4]) ** 3
        # y = 38.12 * torch.atan(x[:, 1]) + -34.37 * torch.atan(torch.sinh(x[:, 1]))
        # y = x[..., 3] * x[..., 4]
        tree = FunctionTree(x, y, torch.nn.MSELoss(),
                            unary_funcs=["abs", "sign", "ceil",
                                         "floor", "log", "exp",
                                         "sqrt", "cos", "sin",
                                         "tanh", "square", "cube", "!"],
                            binary_funcs=["/", "max", "min", "*",
                                          "==", "!=", ">",
                                          "<", "<=", ">=",
                                          r"/\\", r"\/"])

        tree.train(pop_size=200, epochs=2)

        idx = np.argmin([v.n_outliers for v in tree.stls_vars])
        final_node = tree.stls_vars[idx]
        print(final_node.get_name())
        y_hat = final_node.value
        plt.scatter(y, y_hat)
        plt.show()
        _ = [print(v.get_name()) for v in tree.stls_vars]

        # [((y-v.value)**2).max() for v in tree.stls_vars]
        #
        # [((y-v.value)**2).max() for v in tree.stls_vars]
        #
        # e = y_hat.squeeze()-y
        # (abs(e)-2*e.std() > 0).sum()

        # f = BinaryNode("*", 0, BaseNode(x[:1], "x1", 1), BaseNode(x[:2], "x2", 2))
        # f.forward(x).shape
        #
        # final_node.forward(x).shape
        # final_node.x1.x1.x1.x1.forward(x).shape
        #
        # x1 = final_node
        #
        # while x1.forward(x).shape == torch.Size([100, 100]):
        #     x2 = x1
        #     x1 = x1.x1
        #
        # cust = BinaryNode("*", 0,
        #                   ScalarNode(x[..., 1], '-2.75', -2.75),
        #                   UnaryNode("sqrt", 0,
        #                             BaseNode(x[..., 1], "x1", 1)))

        model = tree.get_best()

        y_hat = model.forward(x)

        plt.scatter(y, y_hat)
        plt.show()

        complete_vars = [x for x in tree.all_vars + tree.stls_vars if torch.var(x.value.to(float)) > 0.]

        losses = np.array([[x.loss, x.complexity] for x in complete_vars if x.loss is not None])
        node = complete_vars[np.argmin(losses[:, 0])]

        plt.scatter(np.log(losses[:, 0]), np.log(losses[:, 1]))
        plt.show()



        print("done")
        [print(x.get_name()) for x in tree.all_vars]

    def STLS(self, d, y, threshold=0.1, thresh_inc=0.1, max_thresh=100, n_param_target=5):
        dtmp = d
        idx = np.arange(d.shape[-1])
        clf = linear_model.LinearRegression(fit_intercept=False)
        while dtmp.shape[-1] > n_param_target and threshold < max_thresh:
            n_cmp = -1
            threshold += thresh_inc
            while n_cmp != dtmp.shape[-1]:
                clf.fit(dtmp, y)
                coef = clf.coef_.round(decimals=2)
                y_hat = np.matmul(d[:, idx], coef)
                print(f"Loss:{((y - y_hat) ** 2).mean():.4f}")

                flt = np.abs(coef) > threshold
                idx = idx[flt]
                n_cmp = dtmp.shape[-1]
                dtmp = dtmp[:, flt]
            # if np.min(np.abs(coef)) > 1000:
            #     break
        return coef, idx

    def gradient_descent_SINDy(self, d, y):
        l = torch.nn.Linear(in_features=d.shape[-1], out_features=1, bias=False)
        torch.nn.init.xavier_uniform_(l.weight)
        optimizer = torch.optim.Adam(l.parameters(), lr=1e-3, eps=1e-5)
        lmbda = 0.1
        # with torch.no_grad():
        #     l.weight[-1] = 1
        #     l.weight[:, :-1] = 0
        #     l.bias[:] = 0
        for i in range(10000):
            y_hat = l(d)
            mse_loss = torch.nn.MSELoss()(y, y_hat.squeeze())
            l1_norm = torch.norm(l.weight, p=1)

            non_zero = torch.mean(torch.tanh(torch.abs(l.weight * 100)))
            v = torch.var(y_hat)
            # non_zero = torch.sum(l.weight!=0)

            loss = mse_loss + lmbda * l1_norm - v  # + non_zero * 100
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Loss:{loss:4f}\tNon_zero_weights:{(l.weight == 0).sum().item()}")


if __name__ == '__main__':
    unittest.main()
