import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt

import wandb
from torch import optim, nn

from common.espl import EQL
from helper_local import wandb_login, add_espl_args
from sym import printsymbolic


def run_espl_x_squared(args):
    arch_index = 0
    # init_op_list(arch_index)
    obs_dim = 1
    action_dim = 1
    cfg = dict(
        arch_index=args.arch_index,
        epochs=args.epochs,
        data_size=args.data_size,
        lr=args.lr,
        sample_num=args.sample_num,
        hard_gum=args.hard_gum,
        data_scale=args.data_scale,
        wandb_tags=args.wandb_tags,
        other_loss_scale=args.other_loss_scale,
        hard_ratio=args.hard_ratio,
        dist_func=args.dist_func,
    )
    eql_args = dict(
        target_ratio=args.target_ratio,
        spls=args.spls,
        constrain_scale=args.constrain_scale,
        l0_scale=args.l0_scale,
        bl0_scale=args.bl0_scale,
        target_temp=args.target_temp,
        warmup_epoch=args.warmup_epoch,
        hard_epoch=int(cfg["epochs"] * cfg["hard_ratio"]),
    )
    cfg.update(eql_args)
    epochs = cfg["epochs"]
    data_size = cfg["data_size"]
    lr = cfg["lr"]
    sample_num = cfg["sample_num"]
    hard_gum = cfg["hard_gum"]
    data_scale = cfg["data_scale"]
    other_loss_scale = cfg["other_loss_scale"]

    if args.dist_func == "mse":
        dist_func = nn.MSELoss()
    elif args.dist_func == "maxse":
        dist_func = lambda y, y_hat: ((y - y_hat) ** 2).max()
    elif args.dist_func == "meanmax":
        def dist_func(y, y_hat):
            sq_diff = (y - y_hat) ** 2
            return sq_diff.mean() + sq_diff.max()*0.1
    else:
        raise NotImplementedError(f"dist_func {args.dist_func} not implemented. Must be one of 'mse', 'maxse'")

    num_inputs = obs_dim
    num_outputs = action_dim
    model = EQL(num_inputs, num_outputs, sample_num, hard_gum, **eql_args)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.sample_sparse_constw(mode=0)

    obs = (np.random.random((data_size, 1)) - .5) * data_scale
    x = torch.FloatTensor(obs)
    y = x ** 2

    obs_ood = (np.random.random((data_size, 1)) - .5) * data_scale * 5
    x_ood = torch.FloatTensor(obs_ood)
    y_ood = x_ood ** 2

    if sample_num > 1:
        y = y.unsqueeze(0).expand(sample_num, -1, -1).reshape(sample_num * data_size, -1)

    name = np.random.randint(1e5)
    wandb_login()
    wb_resume = "allow"  # if args.model_file is None else "must"
    prefix = "espl"
    wandb.init(project="espl", config=cfg, sync_tensorboard=True,
               tags=cfg["wandb_tags"], resume=wb_resume, name=f"{prefix}-{name}")

    for epoch in range(epochs):
        y_hat = model.forward(x, mode=1)

        other_loss, sparse_loss, constrain_loss, regu_loss, l0_loss, bl0_loss = model.get_loss()
        dist_loss = dist_func(y, y_hat)

        total_loss = dist_loss + other_loss * other_loss_scale
        model.update_const()

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        model.proj()
        if epoch % (epochs // 100) == 0:
            print(f"Epoch:{epoch}\tLoss:{total_loss.item():.2f}")

        with torch.no_grad():
            model.sample_sparse_constw(mode=0)
            y_ood_hat = model.forward(x_ood, mode=0)
            ood_mse_loss = nn.MSELoss()(y_ood_hat, y_ood)
            sparsity = model.constw_mask.mean()

        if args.dist_func != "mse":
            with torch.no_grad():
                mse_loss = ((y-y_hat)**2).mean()
        else:
            mse_loss = dist_loss
        wandb.log({
            "total_loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            'dist_loss': dist_loss.item(),
            'ood_mse_loss': ood_mse_loss.item(),
            "sparse_loss": sparse_loss.item(),
            "other_loss": other_loss.item(),
            "constrain_loss": constrain_loss.item(),
            "regu_loss": regu_loss.item(),
            "l0_loss": l0_loss.item(),
            "bl0_loss": bl0_loss.item(),
            "epoch": epoch,
            "temp": model.temp,
            "target_ratio": model.target_ratio_current,
            "sparsity": sparsity.item(),
        })
        model.set_temp_target_ratio(epoch)
    zero_weight_pct = (model.constw == 0).sum() / np.prod(model.constw.shape)

    model.sample_sparse_constw(mode=0)
    y_hat_mode_0 = model.forward(x, mode=0)
    zero_weight_pct_mode_0 = (model.constw == 0).sum() / np.prod(model.constw.shape)

    data = [[x, y] for (x, y) in zip(obs.squeeze().tolist(), y_hat.squeeze().detach().cpu().numpy().tolist())]
    table = wandb.Table(data=data, columns=["x", "y_hat"])
    wandb.log({"predicted y_hat": wandb.plot.scatter(table, "x", "y_hat",
                                                     title="Y_hat prediction vs x")})
    data = [[x, y] for (x, y) in zip(obs.squeeze().tolist(), y_hat_mode_0.squeeze().detach().cpu().numpy().tolist())]
    table = wandb.Table(data=data, columns=["x", "y_hat_mode_0"])
    wandb.log({"predicted y_hat_mode_0": wandb.plot.scatter(table, "x", "y_hat_mode_0",
                                                            title="Y_hat_mode_0 prediction vs x")})

    obs = (np.random.random((data_size, 1)) - .5) * data_scale * 5
    x = torch.FloatTensor(obs)
    y = x ** 2
    y_hat_mode_0_large = model.forward(x, mode=0)
    data = [[x, y] for (x, y) in
            zip(obs.squeeze().tolist(), y_hat_mode_0_large.squeeze().detach().cpu().numpy().tolist())]
    table = wandb.Table(data=data, columns=["x", "ood_mode_0"])
    wandb.log({"predicted out of distribution mode 0": wandb.plot.scatter(table, "x", "ood_mode_0",
                                                                          title="ood_mode_0 prediction vs x")})

    ood_loss = nn.MSELoss()(y, y_hat_mode_0_large)
    scores = model.scores.data
    constw_base = model.constw_base.data
    constb = model.constb.data
    constw = constw_base * ((scores > 0.5).float())
    sym_exp = str(printsymbolic(constw, constb, num_inputs, arch_index))

    exp_length = len(sym_exp)
    if exp_length > 1000:
        sym_exp = f"expression length = {exp_length}"

    wandb.log({
        "pct_0_weight": zero_weight_pct.item(),
        "pct_0_weight_mode_0": zero_weight_pct_mode_0.item(),
        "symbolic_expression": str(sym_exp),
        'ood_loss': ood_loss.item(),
        'expression_length': exp_length,
    })
    wandb.finish()
    return


    obs = (np.random.random((data_size, 1)) - .5) * data_scale * 5
    x = torch.FloatTensor(obs)
    y = x ** 2
    mse_best = torch.inf
    for i in range(1000):
        model.sample_sparse_constw(mode=0)
        y_hat = model.forward(x, mode=0)
        mse = ((y - y_hat) ** 2).mean()
        sparsity = model.constw_mask.mean()
        if mse < mse_best:
            y_best = y_hat
            mse_best = mse
            best_sparsity = sparsity
            best_constw_mask = model.constw_mask
        print(f"{i}: sparsity: {sparsity :.2f}\tmse: {mse :,.2f}")
        # plt.scatter(obs.squeeze(), y_hat.squeeze().detach().cpu().numpy(), label=str(i))
    plt.scatter(obs.squeeze(), y.squeeze().detach().cpu().numpy(), label="x**2")
    plt.scatter(obs.squeeze(), y_best.squeeze().detach().cpu().numpy(), label="best")
    plt.show()
    constw = constw_base * best_constw_mask
    sym_exp = str(printsymbolic(constw, constb, num_inputs, arch_index))
    print(sym_exp)

    def sample_sparse_constw(model, eps):
        clamped_scores = torch.sigmoid(model.scores)
        model.constw_mask = (torch.rand_like(model.scores) + eps < clamped_scores).float()
        model.constw = model.constw_base * model.constw_mask

    obs = (np.random.random((data_size, 1)) - .5) * data_scale * 5
    x = torch.FloatTensor(obs)
    y = x ** 2
    mse_best = torch.inf
    for i in range(50):
        eps = i/100
        sample_sparse_constw(model, eps)
        y_hat = model.forward(x, mode=0)
        mse = ((y - y_hat) ** 2).mean()
        sparsity = model.constw_mask.mean()
        if mse < mse_best:
            y_best = y_hat
            mse_best = mse
            sparsity_best = sparsity
            eps_best = eps
        print(f"{i}: sparsity: {sparsity :.2f}\tmse: {mse :,.2f}")
        # plt.scatter(obs.squeeze(), y_hat.squeeze().detach().cpu().numpy(), label=str(i))
    plt.scatter(obs.squeeze(), y.squeeze().detach().cpu().numpy(), label="x**2")
    plt.scatter(obs.squeeze(), y_best.squeeze().detach().cpu().numpy(), label="best")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_espl_args(parser)
    args = parser.parse_args()
    # args.dist_func = "meanmax"
    args.target_temp = 0.01
    # for e in range(1,10):
    #     args.target_temp = 0.1/e
    run_espl_x_squared(args)
