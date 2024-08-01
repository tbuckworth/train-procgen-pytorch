import unittest
import torch
from matplotlib import pyplot as plt

from common import xavier_uniform_init
from symb_reg_torch import create_func, run_tree, BinaryNode


from sklearn import linear_model


class MyTestCase(unittest.TestCase):
    def test_something(self):
        lmbda = 0.1
        x = torch.rand((100, 5))
        # y = torch.cos(x[:, 1])**2 + torch.sin(x[:, 5])**3
        y = x[..., 3] * x[..., 4]
        tree = run_tree(x, y, 200, 50)
        model = tree.get_best()
        vrs = tree.all_vars
        tmp_vars = vrs[:5]
        tmp_vars.append(BinaryNode("*", 1, tmp_vars[3], tmp_vars[4]))

        dictionary = [v.evaluate() for v in tmp_vars]
        d = torch.cat(dictionary, dim=1)

        d = d[:10]
        y = y[:10]

        clf = linear_model.Lasso(alpha=0.1, fit_intercept=False)
        clf.fit(d, y)
        print(clf.coef_)
        print(clf.intercept_)




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

            loss = mse_loss + lmbda * l1_norm -v#+ non_zero * 100
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Loss:{loss:4f}\tNon_zero_weights:{(l.weight == 0).sum().item()}")

        y_hat = model.forward(x)

        plt.scatter(y, y_hat)
        plt.show()

        print("done")
        [print(x.get_name()) for x in tree.all_vars]

if __name__ == '__main__':
    unittest.main()
