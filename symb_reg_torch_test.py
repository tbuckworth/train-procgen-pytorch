import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt

from common import xavier_uniform_init
from symb_reg_torch import create_func, run_tree, BinaryNode

from sklearn import linear_model


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # lmbda = 0.1
        x = torch.rand((100, 5))
        # y = torch.cos(x[:, 1])**2 + torch.sin(x[:, 5])**3
        y = x[..., 3] * x[..., 4]
        tree = run_tree(x, y, 200, 50)
        model = tree.get_best()

        # vrs = tree.all_vars
        # tmp_vars = vrs[:5]
        # tmp_vars.append(BinaryNode("*", 1, tmp_vars[3], tmp_vars[4]))

        dictionary = [v.evaluate() for v in tree.all_vars]
        d = torch.cat(dictionary, dim=1)

        coef, idx = self.STLS(d, y)
        print(coef)
        nms = np.array([v.get_name() for v in tree.all_vars])[idx].tolist()

        function_name = '+'.join([f"{c:.2f}{f}" for c, f in zip(coef, nms)])
        print(function_name)

        y_hat = np.matmul(d[:, idx], coef)


        # print(clf.intercept_)

        # y_hat = model.forward(x)

        plt.scatter(y, y_hat)
        plt.show()

        print("done")
        [print(x.get_name()) for x in tree.all_vars]

    def STLS(self, d, y):
        dtmp = d
        idx = np.arange(d.shape[-1])
        clf = linear_model.LinearRegression(fit_intercept=False)
        threshold = 0.1
        while dtmp.shape[-1] > 5 and threshold < 100:
            n_cmp = -1
            threshold += 0.1
            while n_cmp != dtmp.shape[-1]:
                if dtmp.shape[-1] == 0:
                    print("huh?")
                clf.fit(dtmp, y)
                coef = clf.coef_.round(decimals=2)
                # print(coef)
                y_hat = np.matmul(d[:, idx], coef)
                print(f"Loss:{((y-y_hat)**2).mean():.4f}")

                flt = np.abs(coef) > threshold
                idx = idx[flt]
                n_cmp = dtmp.shape[-1]
                dtmp = dtmp[:, flt]
            if np.min(np.abs(coef)) > 1000:
                break
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
