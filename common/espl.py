import math

import einops
import torch
from torch import nn

# op_list = []
#
# op_index_list = []
# op_index_list.append([0, 0, 0])
# op_index_list.append([1, 1, 1])
# op_index_list.append([2, 2, 3, 3, 4, 4, 5, 5])
# op_index_list.append([2, 2, 3, 3, 4, 4, 5, 5])
# op_index_list.append([0, 1, 2, 3])
# op_index_list.append([0, 1, 2, 3])
# op_list.append(op_index_list)
#
#
# def get_sym_arch(index):
#     return op_list[index]


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


# def init_op_list(index):
#     global op_list
#     global op_regu_list
#     global op_in_list
#     global op_inall_list
#     global op_index_list
#     op_index_list = get_sym_arch(index)
#     op_list = []
#     op_regu_list = []
#     op_in_list = []
#     op_inall_list = []
#     for layer in op_index_list:
#         op = []
#         op_regu = []
#         op_in_num = []
#         op_inall = 0
#         for index in layer:
#             op.append(base_op[index])
#             op_regu.append(regu_op[index])
#             op_in_num.append(op_in[index])
#             op_inall += op_in[index]
#         op_list.append(op)
#         op_regu_list.append(op_regu)
#         op_in_list.append(op_in_num)
#         op_inall_list.append(op_inall)
#
#     print("N inputs for each operator:")
#     print(op_in_list)
#     print(op_inall_list)
#     print(op_index_list)


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
                 arch_index=0,
                 ):
        super(EQL, self).__init__()
        self.init_op_list(arch_index)
        self.target_ratio = target_ratio
        self.target_ratio_current = 1 - target_ratio
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
        self.depth = len(self.op_list)

        wshape = 0
        inshape_ = self.num_inputs
        bshape = 0
        for i in range(self.depth):
            bshape += self.op_inall_list[i]
            wshape += inshape_ * self.op_inall_list[i]
            inshape_ += len(self.op_in_list[i])
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
        self._sample_num = sample_num

    def get_sym_arch(self, index):
        op_list = []

        op_index_list = []
        op_index_list.append([0, 0, 0])
        op_index_list.append([1, 1, 1])
        op_index_list.append([2, 2, 3, 3, 4, 4, 5, 5])
        op_index_list.append([2, 2, 3, 3, 4, 4, 5, 5])
        op_index_list.append([0, 1, 2, 3])
        op_index_list.append([0, 1, 2, 3])
        op_list.append(op_index_list)

        return op_list[index]

    def init_op_list(self, index=0):
        op_index_list = self.get_sym_arch(index)
        self.op_list = []
        self.op_regu_list = []
        self.op_in_list = []
        self.op_inall_list = []
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
            self.op_list.append(op)
            self.op_regu_list.append(op_regu)
            self.op_in_list.append(op_in_num)
            self.op_inall_list.append(op_inall)

    def opfunc(self, input, index, mode):  # input: batch,op_inall
        op_out = []
        regu_loss = 0
        if mode == 1:
            offset = 0
            for i in range(len(self.op_in_list[index])):
                if self.op_in_list[index][i] == 1:
                    out, regu = self.op_regu_list[index][i](input[..., offset])
                    op_out.append(out)
                    regu_loss += regu
                    offset += 1
                elif self.op_in_list[index][i] == 2:
                    out, regu = self.op_regu_list[index][i](input[..., offset], input[..., offset + 1])
                    op_out.append(out)
                    regu_loss += regu
                    offset += 2
                elif self.op_in_list[index][i] == 3:
                    out, regu = self.op_regu_list[index][i](input[..., offset], input[..., offset + 1],
                                                            input[..., offset + 2])
                    op_out.append(out)
                    regu_loss += regu
                    offset += 3
        else:
            offset = 0
            for i in range(len(self.op_in_list[index])):
                if self.op_in_list[index][i] == 1:
                    out = self.op_list[index][i](input[..., offset])
                    op_out.append(out)
                    offset += 1
                elif self.op_in_list[index][i] == 2:
                    out = self.op_list[index][i](input[..., offset], input[..., offset + 1])
                    op_out.append(out)
                    offset += 2
                elif self.op_in_list[index][i] == 3:
                    out = self.op_list[index][i](input[..., offset], input[..., offset + 1], input[..., offset + 2])
                    op_out.append(out)
                    offset += 3
            # print(offset)
        return torch.stack(op_out, dim=-1), regu_loss

    def constrain_loss(self):

        return torch.clamp(self.scores - 6, min=0).sum(-1).sum(-1).mean() + torch.clamp(-6 - self.scores, min=0).sum(
            -1).sum(-1).mean()

    def proj(self):
        # added?
        self.scores.data.clamp_(0, 1)

    def sparse_loss(self):
        clamped_scores = torch.sigmoid(self.scores)
        return torch.clamp(clamped_scores.sum(-1).sum(-1) - self.target_ratio_current * self.wshape * self.num_outputs,
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

    def set_mode(self, mode):
        if self.mode and not mode:
            # going from sample mode to use mode requires re-sampling weights, but not the other way around
            self.sample_sparse_constw(mode)
        else:
            self.mode = mode

    def sample_sparse_constw(self, mode):
        self.mode = mode
        if mode:
            self.sample_num = self._sample_num
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
            self.sample_num = 1
            clamped_scores = torch.sigmoid(self.scores)
            self.constw_mask = (torch.rand_like(self.scores) < clamped_scores).float()
            self.constw = self.constw_base * self.constw_mask

    def forward(self, obs, mode=-1):
        if mode == -1:
            mode = self.mode

        in_shape = obs.shape
        x = obs.reshape(-1, *in_shape[-1:])

        if mode:
            self.sample_sparse_constw(1)
            constw = self.constw
            constb = self.constb.unsqueeze(0).expand(self.sample_num, self.num_outputs, -1)
        else:
            constw = self.constw.unsqueeze(0)
            constb = self.constb

        w_list = []
        inshape_ = self.num_inputs
        low = 0
        for i in range(self.depth):
            high = low + inshape_ * self.op_inall_list[i]
            w_list.append(constw[..., int(low):int(high)])
            inshape_ += len(self.op_in_list[i])
            low = high

        w_last = constw[..., int(low):]

        b_list = []
        low = 0
        for i in range(self.depth):
            high = low + self.op_inall_list[i]
            b_list.append(constb[..., low:high])
            low = high
        b_last = constb[..., low:]

        shp = (self.sample_num, *(-1 for _ in x.shape), self.num_outputs)
        x = x.view(1, *x.shape, 1).expand(shp)
        x = x.transpose(-1, -2)
        # num_samples, batch, outputs, inputs

        reguloss = 0
        inshape_ = self.num_inputs
        for i in range(self.depth):
            w = w_list[i].unsqueeze(-1)
            w = w.reshape(self.sample_num, self.num_outputs, self.op_inall_list[i], inshape_)

            inshape_ += len(self.op_in_list[i])
            b = b_list[i].unsqueeze(-3)

            hidden = einops.einsum(x, w, "num batch out in, num out op_in in -> num batch out op_in") + b

            op_hidden, regu = self.opfunc(hidden, i, mode)
            x = torch.cat([x, op_hidden], dim=-1)
            reguloss += regu
        w = w_last
        # b_last: num out 1 -> num 1 out (cast to num batch out after einops)
        b = b_last.squeeze(-1).unsqueeze(-2)
        try:
            out = einops.einsum(x, w, "num batch out in, num out in -> num batch out") + b
        except RuntimeError as e:
            raise e
        self.regu_loss = reguloss

        out_shape = [self.sample_num] + list(in_shape[:-1]) + [self.num_outputs]
        return out.reshape(out_shape).squeeze()

    def set_temp_target_ratio(self, epoch):
        self.temp = 1 / ((1 - self.target_temp) * (
                1 - min(epoch, self.hard_epoch) / self.hard_epoch) + self.target_temp)
        clip_it = max(min(epoch, self.hard_epoch), self.warmup_epoch)
        self.target_ratio_current = self.target_ratio + (1 - self.target_ratio) * (
                1 - ((clip_it - self.warmup_epoch) / (self.hard_epoch - self.warmup_epoch)) ** 2)
