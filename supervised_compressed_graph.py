import argparse

import torch
from matplotlib import pyplot as plt
from torch import nn

from common.model import CompressedGraph
from double_graph_sr import find_model
from helper_local import create_logdir, create_symb_dir_if_exists, add_symbreg_args
from multiprocessing import Pool



def pool_idea():

    from pysr import PySRRegressor

    def symbolic_regression_on_dataset(data):
        X, y = data["X"], data["y"]
        model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            verbosity=0,
        )
        model.fit(X, y)
        return model

    # Example: List of datasets
    datasets = [
        {"X": X1, "y": y1},
        {"X": X2, "y": y2},
        {"X": X3, "y": y3},
    ]

    with Pool(len(datasets)) as pool:
        models = pool.map(symbolic_regression_on_dataset, datasets)

    # Access models for each dataset
    for i, model in enumerate(models):
        print(f"Dataset {i + 1} symbolic expression: {model}")

def sr_fit(data):
    # X, Y, symbdir, save_file, weights, args
    msg_model, _ = find_model(**data)
    return msg_model.pytorch()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 10
    k = 2
    m = 3
    b = int(args.epochs / (m ** 2))
    lr = 1e-3
    epochs = args.epochs
    seed = 42
    ensemble = False
    folder = "test"
    project = "compressed_graph"
    subfolder = "supervised"
    logdir = create_logdir(seed, folder, project, subfolder)
    symbdir, save_file = create_symb_dir_if_exists(logdir)

    x = torch.rand((b, n)).to(device)
    y = generate_y(x, (b, k)).to(device)
    x_test = x * 2
    y_test = generate_y(x_test, (b, k)).to(device)

    model = CompressedGraph(in_channels=n,
                            action_size=k,
                            compressed_size=m,
                            depth=4,
                            mid_weight=64,
                            device=device)
    neural_train_loss = train_and_save(model, x, y, epochs, lr, logdir + "/neural_model.pth", nn.MSELoss())
    y_hat_neural = model(x)
    y_test_hat_neural = model(x_test)
    neural_test_loss = (y_test - y_test_hat_neural).pow(2).mean()
    # Train the pysr bit
    weights = None
    with torch.no_grad():
        m_in, m_out = model.forward_for_imitation(x)

    datasets = [
        dict(X=m_in[:,i//m, int(i/m),(0,2)].detach().cpu().numpy(),
             Y=m_out[:,i//m, int(i/m)].detach().cpu().numpy(),
             symbdir=symbdir,
             save_file=save_file,
             weights=weights,
             args=args) for i in range(m**2)
    ]

    # with Pool(len(datasets)) as pool:
    #     models = pool.map(sr_fit, datasets)

    models = [sr_fit(data) for data in datasets]

    model.assign_symb_models(models)
    symb_loss = nn.MSELoss()

    # if old_school_method:
        # sr_x = flatten_batches_to_numpy(m_in)
        # sr_y = flatten_batches_to_numpy(m_out).squeeze()
        # msg_model, _ = find_model(sr_x, sr_y, symbdir, save_file, weights, args)
        # if ensemble:
        #     msg_torch = all_pysr_pytorch(msg_model, device)
        #     symb_loss = min_batch_loss
        # else:
        #     msg_torch = NBatchPySRTorch(msg_model.pytorch(), device)
        #     symb_loss = nn.MSELoss()

        # Fine tune sr bit:
        # model.messenger = msg_torch



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
    parser = argparse.ArgumentParser()
    parser = add_symbreg_args(parser)
    args = parser.parse_args()

    args.epochs = 5000
    args.verbosity = 1
    args.iterations = 5
    args.populations = 25
    args.procs = 8
    args.n_cycles_per_iteration = 4000
    args.denoise = False
    main(args)
