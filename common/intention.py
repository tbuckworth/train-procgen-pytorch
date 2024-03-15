import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch import jit

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
        itn = self.get_itn_weights(k, v)
        h = einops.einsum(queries, itn, "b f d2, b d2 d1 -> b f d1")
        return h

    def get_itn_weights(self, k, v):
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
        # Optionally add skip connection to kkikv...?
        if self.sigma:
            itn = self.softmax(kkikv)
        else:
            itn = kkikv
        return itn

    def forward_plus_itn(self, q, k, v):
        queries = self.query(q)
        itn = self.get_itn_weights(k, v)
        h = einops.einsum(queries, itn, "b f d2, b d2 d1 -> b d1")
        return h, itn


class SelfIntention(nn.Module):
    def __init__(self, input_dim, alpha=0.001, sigma=True):
        super(SelfIntention, self).__init__()
        self.intention = Intention(input_dim, alpha, sigma)

    def forward(self, x):
        return self.intention.forward(x, x, x)

    def get_itn_weights(self, x):
        return self.intention.get_itn_weights(x, x)

    def forward_plus_itn(self, x):
        return self.intention.forward_plus_itn(x, x, x)


class MultiHeadIntention(nn.Module):
    #TODO: technically we should reduce the dimensions for the individual attention heads
    # e.g.:
    # head_dim = self.embed_dim // self.num_heads
    # assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
    # scaling = float(head_dim) ** -0.5

    def __init__(self, input_dim, num_heads, alpha=0.001, sigma=True):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfIntention(input_dim, alpha, sigma)
             for _ in range(num_heads)]
        )
        self.w0 = nn.Linear(input_dim*num_heads, input_dim)
        #TODO: what about skip connection and layernorm?
    def forward(self, x):
        all_heads = torch.cat([head(x) for head in self.heads], dim=-1)
        y = self.w0(all_heads)
        return y

    def get_itn_weights(self, x):
        w = [head.get_itn_weights(x) for head in self.heads]
        return einops.rearrange(w, "n b d0 d1 -> b n d0 d1")

    def get_attn_weights(self, x):
        return self.get_itn_weights(x)

    def forward_plus_attn_weights(self, x):
        w = [head.forward_plus_itn(x) for head in self.heads]
        all_heads = torch.cat(w, dim=-1)
        out = self.w0(all_heads)
        itn_weights = einops.rearrange(w, "n b d0 d1 -> b n d0 d1")
        return out, itn_weights







if __name__ == "__main__":
    x = torch.randn(640 * 2).reshape((2, 64, 10))

    mhi = MultiHeadIntention(input_dim=10, num_heads=8)
    #
    y = mhi(x)
    print(x.shape)
    print(y.shape)

    intention = SelfIntention(input_dim=10)
    print(intention.get_itn_weights(x).shape)
    print(intention(x).shape)

    # mha = nn.MultiheadAttention(embed_dim=10, num_heads=5)
    # y2, _ = mha(x, x, x)
    # print(y2.shape)
