import math

import numpy as np
import torch
import wandb
from torch import nn, optim
from torchinfo import summary
import matplotlib.pyplot as plt

from helper_local import wandb_login

op_list = []

op_index_list = []
op_index_list.append([0, 0, 0])
op_index_list.append([1, 1, 1])
op_index_list.append([2, 2, 3, 3, 4, 4, 5, 5])
op_index_list.append([2, 2, 3, 3, 4, 4, 5, 5])
op_index_list.append([0, 1, 2, 3])
op_index_list.append([0, 1, 2, 3])
op_list.append(op_index_list)


def get_sym_arch(index):
    return op_list[index]


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def identity_regu(input):
    return input, 0


def lim_log_regu(input):
    return torch.log(torch.clamp(input, min=0.001)), torch.clamp(0.001 - input, min=0).mean()


def lim_exp_regu(input):
    return torch.exp(torch.clamp(input, -10, 4)), (
            torch.clamp(-10 - input, min=0) + torch.clamp(input - 4, min=0)).mean()


def lim_second_regu(input):
    return torch.clamp(input, min=-20, max=20) ** 2, (
            torch.clamp(-20 - input, min=0) + torch.clamp(input - 20, min=0)).mean()


def lim_third_regu(input):
    return torch.clamp(input, min=-10, max=10) ** 3, (
            torch.clamp(-10 - input, min=0) + torch.clamp(input - 10, min=0)).mean()


def lim_sqrt_regu(input):
    return torch.sqrt(torch.clamp(input, min=0.0001)), (torch.clamp(0.0001 - input, min=0)).mean()


def lim_div_regu(input1, input2):
    output = input1 / input2.masked_fill(input2 < 0.01, 0.01)
    output = output.masked_fill(input2 < 0.01, 0.0)
    return output, torch.clamp(0.01 - input2, min=0).mean()


def mul_regu(input1, input2):
    return torch.clamp(input1, min=-100, max=100) * torch.clamp(input2, min=-100, max=100), (
            torch.clamp(-100 - input1, min=0) + torch.clamp(input1 - 100, min=0)).mean() + (
                   torch.clamp(-100 - input2, min=0) + torch.clamp(input2 - 100, min=0)).mean()


def cos_regu(input):
    return torch.cos(input), 0


def sin_regu(input):
    return torch.sin(input), 0


def identity(input):
    return input


def lim_log(input):
    return torch.log(torch.clamp(input, min=0.001))


def lim_exp(input):
    return torch.exp(torch.clamp(input, -10, 4))


def lim_second(input):
    return torch.clamp(input, min=-20, max=20) ** 2


def lim_third(input):
    return torch.clamp(input, min=-10, max=10) ** 3


def lim_sqrt(input):
    return torch.sqrt(torch.clamp(input, min=0.0001))


def lim_div(input1, input2):
    output = input1 / input2.masked_fill(input2 < 0.01, 0.01)
    output = output.masked_fill(input2 < 0.01, 0.0)
    return output


def mul(input1, input2):
    return torch.clamp(input1, min=-100, max=100) * torch.clamp(input2, min=-100, max=100)


def ifelse(condinput, input1, input2):
    cond = torch.sigmoid(condinput)
    return cond * input1 + (1 - cond) * input2


def ifelse_regu(condinput, input1, input2):
    cond = torch.sigmoid(condinput)
    return cond * input1 + (1 - cond) * input2, 0


base_op = [mul, lim_div, lim_log, lim_exp, torch.sin, torch.cos, identity, ifelse]
regu_op = [mul_regu, lim_div_regu, lim_log_regu, lim_exp_regu, sin_regu, cos_regu, identity_regu, ifelse_regu]
op_in = [2, 2, 1, 1, 1, 1, 1, 3]


def init_op_list(index):
    global op_list
    global op_regu_list
    global op_in_list
    global op_inall_list
    global op_index_list
    op_index_list = get_sym_arch(index)
    op_list = []
    op_regu_list = []
    op_in_list = []
    op_inall_list = []
    for layer in op_index_list:
        op = []
        op_regu = []
        op_in_num = []
        op_inall = 0
        for index in layer:
            op.append(base_op[index])
            op_regu.append(regu_op[index])
            op_in_num.append(op_in[index])
            op_inall += op_in[index]
        op_list.append(op)
        op_regu_list.append(op_regu)
        op_in_list.append(op_in_num)
        op_inall_list.append(op_inall)

    print("N inputs for each operator:")
    print(op_in_list)
    print(op_inall_list)
    print(op_index_list)


def opfunc(input, index, mode):  # input: batch,op_inall
    op_out = []
    regu_loss = 0
    if mode == 1:
        offset = 0
        for i in range(len(op_in_list[index])):
            if op_in_list[index][i] == 1:
                out, regu = op_regu_list[index][i](input[:, offset])
                op_out.append(out)
                regu_loss += regu
                offset += 1
            elif op_in_list[index][i] == 2:
                out, regu = op_regu_list[index][i](input[:, offset], input[:, offset + 1])
                op_out.append(out)
                regu_loss += regu
                offset += 2
            elif op_in_list[index][i] == 3:
                out, regu = op_regu_list[index][i](input[:, offset], input[:, offset + 1], input[:, offset + 2])
                op_out.append(out)
                regu_loss += regu
                offset += 3
    else:
        offset = 0
        for i in range(len(op_in_list[index])):
            if op_in_list[index][i] == 1:
                out = op_list[index][i](input[:, offset])
                op_out.append(out)
                offset += 1
            elif op_in_list[index][i] == 2:
                out = op_list[index][i](input[:, offset], input[:, offset + 1])
                op_out.append(out)
                offset += 2
            elif op_in_list[index][i] == 3:
                out = op_list[index][i](input[:, offset], input[:, offset + 1], input[:, offset + 2])
                op_out.append(out)
                offset += 3
        # print(offset)
    return torch.stack(op_out, dim=1), regu_loss


class EQL(nn.Module):
    '''
    Code from ESPL paper (Efficient Symbolic Policy Learning).
    '''

    def __init__(self, num_inputs, num_outputs, sample_num, hard_gum,
                 target_ratio=0.001,
                 spls=0.1,
                 constrain_scale=1,
                 l0_scale=0.01,
                 bl0_scale=0,
                 target_temp=0.2,
                 warmup_epoch=0,
                 hard_epoch=900,
                 ):
        super(EQL, self).__init__()
        self.target_ratio = target_ratio
        self.spls = spls
        self.constrain_scale = constrain_scale
        self.l0_scale = l0_scale
        self.bl0_scale = bl0_scale
        self.target_temp = target_temp
        self.warmup_epoch = warmup_epoch
        self.hard_epoch = hard_epoch

        self.num_inputs = num_inputs

        self.num_outputs = num_outputs

        self.hard_gum = hard_gum
        self.temp = 0.03

        self.repeat = 1
        self.depth = len(op_list)

        wshape = 0
        inshape_ = self.num_inputs
        bshape = 0
        for i in range(self.depth):
            bshape += op_inall_list[i]
            wshape += inshape_ * op_inall_list[i]
            inshape_ += len(op_in_list[i])
        wshape += inshape_
        bshape += 1
        self.wshape = wshape
        self.bshape = bshape

        self.scores = nn.Parameter(torch.Tensor(num_outputs, wshape))
        self.scores.data.fill_(3.0)
        self.constw_base = nn.Parameter(
            torch.Tensor(self.num_outputs, wshape)
        )

        self.constb = nn.Parameter(torch.Tensor(self.num_outputs, bshape))
        bound = 1 / math.sqrt(10)
        self.constw_base.data.uniform_(-bound, bound)
        self.constb.data.uniform_(-bound, bound)

        self.batch = 1
        self.sample_num = sample_num

    def constrain_loss(self):

        return torch.clamp(self.scores - 6, min=0).sum(-1).sum(-1).mean() + torch.clamp(-6 - self.scores, min=0).sum(
            -1).sum(-1).mean()

    def proj(self):
        # added?
        self.scores.data.clamp_(0, 1)

    def sparse_loss(self):
        clamped_scores = torch.sigmoid(self.scores)
        return torch.clamp(clamped_scores.sum(-1).sum(-1) - self.target_ratio * self.wshape * self.num_outputs,
                           min=0).mean() / self.num_outputs

    # def sim_loss(self):
    #     meta_batch  = self.scores.shape[0]
    #     idx= torch.randperm(meta_batch)
    #     shuffle_scores = self.scores[idx,:,:].detach()
    #     return torch.abs(self.scores-shuffle_scores).mean()

    def score_std(self):
        return torch.std(torch.sigmoid(self.scores), dim=0).mean()

    def l0_loss(self):
        return torch.abs(self.constw_base).mean()

    def bl0_loss(self):
        return torch.abs(self.constb).mean()

    def expect_w(self):
        clamped_scores = torch.sigmoid(self.scores)
        return clamped_scores.sum(-1).mean()

    def update_const(self):

        self.sample_sparse_constw(0)

    def get_loss(self):
        sparse_loss = self.sparse_loss()
        constrain_loss = self.constrain_loss()
        l0_loss = self.l0_loss()
        bl0_loss = self.bl0_loss()
        return self.spls * sparse_loss + constrain_loss * self.constrain_scale + self.regu_loss + self.l0_scale * l0_loss + self.bl0_scale * bl0_loss, sparse_loss, constrain_loss, self.regu_loss, l0_loss, bl0_loss

    def sample_sparse_constw(self, mode):

        if mode:
            eps = 1e-20
            scores = self.scores.unsqueeze(0).expand(self.sample_num, -1, -1)
            uniform0 = torch.rand_like(scores)
            uniform1 = torch.rand_like(scores)
            noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
            clamped_scores = torch.sigmoid(scores)
            self.constw_mask = torch.sigmoid(
                (torch.log(clamped_scores + eps) - torch.log(1.0 - clamped_scores + eps) + noise) * self.temp)
            if self.hard_gum:
                hard_mask = torch.where(self.constw_mask > 0.5, torch.ones_like(self.constw_mask),
                                        torch.zeros_like(self.constw_mask))
                constw_base = self.constw_base.unsqueeze(0).expand(self.sample_num, -1, -1)
                self.constw = constw_base * (hard_mask - self.constw_mask.detach() + self.constw_mask)
            else:
                self.constw = self.constw_base * self.constw_mask
        else:
            clamped_scores = torch.sigmoid(self.scores)
            self.constw_mask = (torch.rand_like(self.scores) < clamped_scores).float()
            self.constw = self.constw_base * self.constw_mask

    def forward(self, obs, mode=0):

        batch = obs.shape[0]
        x = obs

        if mode:
            self.sample_sparse_constw(1)
            # constw sample,num_outputs,wshape
            constw = self.constw.unsqueeze(1).expand(-1, batch, -1, -1).reshape(-1, self.num_outputs, self.wshape)
            constb = self.constb.unsqueeze(0).unsqueeze(1). \
                expand(self.sample_num, batch, -1, -1). \
                reshape(-1, self.num_outputs, self.bshape)
        else:
            constw = self.constw.unsqueeze(0).expand(batch, -1, -1)
            constb = self.constb.unsqueeze(0).expand(batch, -1, -1)

        w_list = []
        inshape_ = self.num_inputs
        low = 0
        # shapes:
        # np.cumsum([self.num_inputs] + [len(a) for a in op_in_list])[:-1]*np.array(op_inall_list)
        for i in range(self.depth):
            high = low + inshape_ * op_inall_list[i]
            w_list.append(constw[:, :, int(low):int(high)])
            inshape_ += len(op_in_list[i])
            low = high

        w_last = constw[:, :, int(low):]

        b_list = []
        low = 0
        for i in range(self.depth):
            high = low + op_inall_list[i]
            b_list.append(constb[:, :, low:high])
            low = high
        b_last = constb[:, :, low:]
        # x meta_batch*batch_size,num_inputs
        if mode:
            x = x.unsqueeze(0).unsqueeze(2).expand(self.sample_num, -1, self.num_outputs,
                                                   -1)  # sample_num,batch,num_outputs,num_inputs
            x = x.reshape(self.sample_num * batch * self.num_outputs, self.num_inputs)
        else:
            x = x.unsqueeze(1).expand(-1, self.num_outputs, -1)  # batch,num_outputs,num_inputs
            x = x.reshape(batch * self.num_outputs, self.num_inputs)
        reguloss = 0
        inshape_ = self.num_inputs
        if mode:
            batch = self.sample_num * batch
        for i in range(self.depth):
            w = w_list[i].reshape(batch * self.num_outputs, op_inall_list[i], inshape_)
            inshape_ += len(op_in_list[i])
            # if mode:
            #     print(b_list[i].shape,batch,self.num_outputs,op_inall_list[i])
            b = b_list[i].reshape(batch * self.num_outputs, op_inall_list[i])
            # print(w.shape,x.shape)
            hidden = torch.bmm(w, x.unsqueeze(2)).squeeze(-1) + b
            # print(hidden.shape,i,len(op_in_list[i]),op_inall_list[i])
            op_hidden, regu = opfunc(hidden, i, mode)
            x = torch.cat([x, op_hidden], dim=-1)
            # print(x.shape)
            reguloss += regu
        # print(self.num_inputs+(self.depth-1)*op_num*self.repeat)
        w = w_last.reshape(batch * self.num_outputs, 1, inshape_)
        # print(constb.shape,b_last.shape)
        b = b_last.reshape(batch * self.num_outputs, 1)
        # print(w.shape,x.shape,b.shape)
        out = torch.bmm(w, x.unsqueeze(2)).squeeze(-1) + b
        self.regu_loss = reguloss
        return out.reshape(batch, self.num_outputs)

    def set_temp_target_ratio(self, epoch):
        self.temp = 1 / ((1 - self.target_temp) * (
                1 - min(epoch, self.hard_epoch) / self.hard_epoch) + self.target_temp)
        clip_it = max(min(epoch, self.hard_epoch), self.warmup_epoch)
        self.target_ratio = self.target_ratio + (1 - self.target_ratio) * (
                1 - ((clip_it - self.warmup_epoch) / (self.hard_epoch - self.warmup_epoch)) ** 2)


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
    model = EQL(num_inputs, num_outputs, sample_num, hard_gum,**eql_args)
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
