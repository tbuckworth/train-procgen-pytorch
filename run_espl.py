import numpy as np
import torch
import wandb
from torch import optim, nn

from common.espl import init_op_list, EQL
from helper_local import wandb_login

if __name__ == "__main__":
    op_list_selection = 0
    init_op_list(op_list_selection)
    obs_dim = 1
    action_dim = 1
    cfg = dict(
        op_list_selection=op_list_selection,
        epochs=3000,
        data_size=1000,
        lr=1e-3,
        sample_num=1,
        hard_gum=True,
        data_scale=100,
        wandb_tags=["x_squared"],
    )
    eql_args = dict(
        target_ratio=0.001,
        spls=0.1,
        constrain_scale=1,
        l0_scale=0.01,
        bl0_scale=0,
        target_temp=0.2,
        warmup_epoch=0,
        hard_epoch=900,
    )
    cfg.update(eql_args)
    epochs = cfg["epochs"]
    data_size = cfg["data_size"]
    lr = cfg["lr"]
    sample_num = cfg["sample_num"]
    hard_gum = cfg["hard_gum"]
    data_scale = cfg["data_scale"]

    num_inputs = obs_dim
    num_outputs = action_dim
    model = EQL(num_inputs, num_outputs, sample_num, hard_gum, **eql_args)
    # target_ratio=cfg["target_ratio"],
    # spls=cfg["spls"],
    # constrain_scale=cfg["constrain_scale"],
    # l0_scale=cfg["l0_scale"],
    # bl0_scale=cfg["bl0_scale"],
    # target_temp=cfg["target_temp"],
    # warmup_epoch=cfg["warmup_epoch"],
    # hard_epoch=cfg["hard_epoch"],
    # )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.sample_sparse_constw(mode=0)
    obs = (np.random.random((data_size, 1)) - .5) * data_scale
    x = torch.FloatTensor(obs)
    y = x ** 2

    name = np.random.randint(1e5)
    wandb_login()
    wb_resume = "allow"  # if args.model_file is None else "must"
    prefix = "espl"
    wandb.init(project="espl", config=cfg, sync_tensorboard=True,
               tags=cfg["wandb_tags"], resume=wb_resume, name=f"{prefix}-{name}")

    for epoch in range(epochs):
        y_hat = model.forward(x, mode=1)

        other_loss, sparse_loss, constrain_loss, regu_loss, l0_loss, bl0_loss = model.get_loss()
        total_loss = nn.MSELoss()(y, y_hat) + other_loss
        model.update_const()

        print(total_loss.item())
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        model.proj()
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
            "target_ratio": model.target_ratio,
        })
        model.set_temp_target_ratio(epoch)
    # plt.scatter(y.detach().numpy(), y_hat.detach().numpy())
    zero_weight_pct = (model.constw == 0).sum() / np.prod(model.constw.shape)

    model.sample_sparse_constw(mode=0)
    y_hat_mode_0 = model.forward(x, mode=0)
    zero_weight_pct_mode_0 = (model.constw == 0).sum() / np.prod(model.constw.shape)

    wandb.log({
        "y": y.detach().cpu().numpy(),
        "y_hat": y_hat.detach().cpu().numpy(),
        "y_hat_mode_0": y_hat_mode_0.detach().cpu().numpy(),
        "x": obs,
        "pct_0_weight": zero_weight_pct.item(),
        "pct_0_weight_mode_0": zero_weight_pct_mode_0.item(),
    })
    wandb.finish()
    exit(0)

    plt.scatter(y.detach().numpy(), y_hat.detach().numpy())
    plt.show()

    plt.scatter(obs, y_hat.detach().numpy())
    plt.scatter(obs, y.detach().numpy())
    plt.show()

    print("done")

    # self.target_entropy = -np.prod(
    #     self.env.action_space.shape).item()
    # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    # alpha = self.log_alphs.exp()
    # self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
    # self.alpha_optimizer = optimizer_class(
    #     [self.log_alpha],
    #     lr=policy_lr,
    # )
    # policy_loss = (alpha * log_pi - q_new_actions).mean() + other_loss

    summary(model, input_size=obs.shape)
    print("done")
