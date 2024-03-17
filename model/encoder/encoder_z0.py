import numpy as np
import torch
import torch.nn as nn


class EncodeZ0(nn.Module):
    def __init__(self, emb_dim: int):
        super(EncodeZ0, self).__init__()
        self.lins = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2),
                                  nn.ReLU(), nn.Linear(emb_dim // 2, 2 * emb_dim))

    def forward(self, emb: torch.Tensor):
        h_out = self.lins(emb)
        mean, std = self.split_mean_mu(h_out)
        std= std.abs()
        return mean,std

    def split_mean_mu(self, h):
        last_dim = h.size()[-1] // 2
        res = h[:, :last_dim], h[:, last_dim:]
        return res
