import argparse
import functools as ft
import torch
from matplotlib import pyplot as plt
from torch import nn

from common.model import ExpandedGraph
from helper_local import create_logdir, add_symbreg_args, DictToArgs


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)

    return fn_


tuple_functions = {

}
binary_functions = {
    "*": _reduce(torch.mul),
    "+": _reduce(torch.add),
    "/": torch.div,
    "max": torch.max,
    "min": torch.min,
    "mod": torch.remainder,
    "atan2": torch.atan2,
    "==": torch.eq,
    "!=": torch.ne,
    ">": torch.gt,
    "<": torch.lt,
    "<=": torch.le,
    ">=": torch.ge,
    r"/\\": torch.logical_and,
    r"\/": torch.logical_or,
}

unary_functions = {
    "!": torch.logical_not,
    "abs": torch.abs,
    "sign": torch.sign,
    # Note: May raise error for ints.
    "ceil": torch.ceil,
    "floor": torch.floor,
    "log": torch.log,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "cos": torch.cos,
    "acos": torch.acos,
    "sin": torch.sin,
    "asin": torch.asin,
    "tan": torch.tan,
    "atan": torch.atan,
    # Note: May give NaN for complex results.
    "cosh": torch.cosh,
    "acosh": torch.acosh,
    "sinh": torch.sinh,
    "asinh": torch.asinh,
    "tanh": torch.tanh,
    "atanh": torch.atanh,
    "square": torch.square,
    "cube": lambda x: torch.pow(x, 3),
    "relu": torch.relu,
}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 10
    k = 2
    m = 100
    b = 10#int(1000 / (m ** 2))
    lr = 1e-3
    epochs = args.epochs
    seed = 42
    folder = "test"
    project = "expanded_graph"
    subfolder = "supervised"
    logdir = create_logdir(seed, folder, project, subfolder)

    x = torch.rand((b, n)).to(device)
    y = generate_y(x, (b, k)).to(device)
    x_test = x * 2
    y_test = generate_y(x_test, (b, k)).to(device)

    all_funcs = tuple_functions.copy()
    all_funcs.update(
        {k: lambda x: v(x[0]) for k, v in unary_functions.items()}
    )
    all_funcs.update(
        {k: lambda x: v(x[0],x[1]) for k, v in binary_functions.items()}
    )
    model = ExpandedGraph(in_channels=n,
                          action_size=k,
                          expanded_size=m,
                          binary_funcs=all_funcs,
                          device=device)
    neural_train_loss = train_and_save(model, x, y, epochs, lr, logdir + "/neural_model.pth", nn.MSELoss())
    y_hat_neural = model(x)
    y_test_hat_neural = model(x_test)
    neural_test_loss = (y_test - y_test_hat_neural).pow(2).mean()
    # Train the pysr bit

    symb_loss = nn.MSELoss()
    plot_results(y, y_hat_neural, y_hat_neural, y_test, y_test_hat_neural, y_test_hat_neural, "Duplicates")


    y_hat_symb = model(x)
    symb_train_loss = symb_loss(y_hat_symb, y)
    y_test_hat_symb = model(x_test)
    symb_test_loss = symb_loss(y_test_hat_symb, y_test)

    print(f"Neural Train Loss: {neural_train_loss:.4f}\tTest Loss: {neural_test_loss:.4f}")
    print(f"Symbol Train Loss: {symb_train_loss:.4f}\tTest Loss: {symb_test_loss:.4f}")

    plot_results(y, y_hat_neural, y_hat_symb, y_test, y_test_hat_neural, y_test_hat_symb, "Pre Finetune")

    symb_train_loss = train_and_save(model, x, y, epochs, lr, logdir + "/symb_model.pth", symb_loss)
    y_hat_symb = model(x)
    y_test_hat_symb = model(x_test)
    symb_test_loss = symb_loss(y_test_hat_symb, y_test)

    print(f"Neural Train Loss: {neural_train_loss:.4f}\tTest Loss: {neural_test_loss:.4f}")
    print(f"Symbol Train Loss: {symb_train_loss:.4f}\tTest Loss: {symb_test_loss:.4f}")

    plot_results(y, y_hat_neural, y_hat_symb, y_test, y_test_hat_neural, y_test_hat_symb, "Post Finetune")

    print("done")


def try_funcs(all_funcs):
    inputs = (torch.FloatTensor([0.1, 0.3]), torch.FloatTensor([-0.21, 0.35]))
    for k, v in all_funcs.items():
        try:
            v(inputs)
        except Exception as e:
            print(k)


def plot_results(y, y_hat_neural, y_hat_symb, y_test, y_test_hat_neural, y_test_hat_symb, title):
    # if ensemble:
    #     elite_idx = min_batch_loss(y_hat_symb, y, True)
    #     np_y_train_hat_symb = y_hat_symb[elite_idx].detach().cpu().numpy().reshape(-1)
    #     np_y_test_hat_symb = y_test_hat_symb[elite_idx].detach().cpu().numpy().reshape(-1)
    # else:
    np_y_train_hat_symb = y_hat_symb.detach().cpu().numpy().reshape(-1)
    np_y_test_hat_symb = y_test_hat_symb.detach().cpu().numpy().reshape(-1)
    gold = [y.cpu().numpy().reshape(-1), y_test.cpu().numpy().reshape(-1)]
    pred = [
        [y_hat_neural.detach().cpu().numpy().reshape(-1),
         np_y_train_hat_symb],
        [y_test_hat_neural.detach().cpu().numpy().reshape(-1),
         np_y_test_hat_symb],
    ]
    titles = [["Neural Train", "Symbolic Train"], ["Neural Test", "Symbolic Test"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for row in range(2):
        for col in range(2):
            axes[row][col].scatter(
                x=gold[row],
                y=gold[row],
                c='black',
            )
            axes[row][col].scatter(
                x=gold[row],
                y=pred[row][col],
            )
            axes[row][col].set_title(titles[row][col])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def min_batch_loss(y_hat, y, return_idx=False):
    y = y.unsqueeze(0)
    losses = ((y_hat - y) ** 2).mean(dim=tuple(range(1, y.ndim)))
    if return_idx:
        return losses.argmin()
    loss = losses.min()
    return loss


def train_and_save(model, x, y, epochs, lr, logdir, loss_fn):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nan
    for epoch in range(epochs):
        y_hat = model(x)
        # loss = ((y_hat - y) ** 2).mean()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f"loss: {loss.item():.4f}")
    # torch.save(model, logdir)
    return loss.item()


def generate_y(x, shp):
    y = torch.zeros(shp)
    y[..., 0] = x[..., 0] * x[..., 4] + torch.sin(x[..., 2])
    y[..., 1] = x[..., 1] + torch.cos(x[..., 4]) * torch.exp(x[..., 2])
    return y


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser = add_symbreg_args(parser)
    # args = parser.parse_args()
    args = DictToArgs(dict(epochs=1000))
    main(args)
