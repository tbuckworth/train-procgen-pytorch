import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch import jit


class SelfAttention(nn.Module):
    def __init__(self, input_dim, alpha=0.001):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        self.alpha = alpha

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted




class Intention(nn.Module):
    sigma: jit.Final[bool]
    def __init__(self, input_dim, alpha=0.001, sigma=True):
        super(Intention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        self.alpha = alpha
        self.sigma = sigma
    def forward(self, q, k, v):
        queries = self.query(q)
        keys = self.key(k)
        values = self.value(v)

        # K*transpose(K)
        key_sym = einops.einsum(keys, keys, "b f d1, b f d2 -> b d1 d2")

        # + alpha*I
        alpha = torch.Tensor([self.alpha for i in range(self.input_dim)])
        key_sym += torch.diag(alpha)

        # inverted
        key_sym_inv = torch.inverse(key_sym)

        # * transpose(K)
        kkik = einops.einsum(key_sym_inv, keys, "b d0 d1, b f d1 -> b d0 f")

        # * V
        kkikv = einops.einsum(kkik, values, "b d0 f, b f d1 -> b d0 d1")

        # Optionally add skip connection to kkikv

        if self.sigma:
            itn = self.softmax(kkikv)
        else:
            itn = kkikv
        h = einops.einsum(queries, itn, "b f d2, b d2 d1 -> b d1")
        return h

class SelfIntention(nn.Module):
    def __init__(self, input_dim, alpha=0.001, sigma=True):
        super(SelfIntention, self).__init__()
        self.intention = Intention(input_dim, alpha, sigma)

    def forward(self, x):
        return self.intention.forward(x, x, x)


if __name__ == "__main__":

    x = torch.randn(640*2).reshape((2, 64, 10))

    intention = SelfIntention(input_dim=10)

    intention(x)
