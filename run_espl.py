import argparse

import numpy as np
import torch
import wandb
from torch import optim, nn

from common.espl import init_op_list, EQL
from helper_local import wandb_login, add_espl_args
from sym import printsymbolic



def run_espl_x_squared(args):
    arch_index = 0
    init_op_list(arch_index)
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
    )
    eql_args = dict(
        target_ratio=args.target_ratio,
        spls=args.spls,
        constrain_scale=args.constrain_scale,
        l0_scale=args.l0_scale,
        bl0_scale=args.bl0_scale,
        target_temp=args.target_temp,
        warmup_epoch=args.warmup_epoch,
        hard_epoch=int(cfg["epochs"]*cfg["hard_ratio"]),
    )
    cfg.update(eql_args)
    epochs = cfg["epochs"]
    data_size = cfg["data_size"]
    lr = cfg["lr"]
    sample_num = cfg["sample_num"]
    hard_gum = cfg["hard_gum"]
    data_scale = cfg["data_scale"]
    other_loss_scale = cfg["other_loss_scale"]

    num_inputs = obs_dim
    num_outputs = action_dim
    model = EQL(num_inputs, num_outputs, sample_num, hard_gum, **eql_args)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.sample_sparse_constw(mode=0)
    obs = (np.random.random((data_size, 1)) - .5) * data_scale
    x = torch.FloatTensor(obs)
    y = x ** 2
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
        total_loss = nn.MSELoss()(y, y_hat) + other_loss * other_loss_scale
        model.update_const()

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        model.proj()
        if epoch % epochs//100 == 0:
            print(f"Epoch:{epoch}\tLoss:{total_loss.item():.2f}")
        wandb.log({
            "total_loss": total_loss.item(),
            "mse_loss": (total_loss - other_loss).item(),
            "sparse_loss": sparse_loss.item(),
            "other_loss": other_loss.item(),
            "constrain_loss": constrain_loss.item(),
            "regu_loss": regu_loss.item(),
            "l0_loss": l0_loss.item(),
            "bl0_loss": bl0_loss.item(),
            "epoch": epoch,
            "temp": model.temp,
            "target_ratio": model.target_ratio_current,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_espl_args(parser)
    args = parser.parse_args()
    run_espl_x_squared(args)