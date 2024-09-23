import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange, repeat
import numpy as np
from .dilated_conv import DilatedConvEncoder

def D_matrix(N):
    D = torch.zeros(N - 1, N)
    D[:, 1:] = torch.eye(N - 1)
    D[:, :-1] -= torch.eye(N - 1)
    return D


class hp_filter(nn.Module):
    """
        Hodrick Prescott Filter to decompose the series
    """

    def __init__(self, lamb,device):
        super(hp_filter, self).__init__()
        self.lamb = lamb
        self.device=device

    def forward(self, x):
        x = x.permute(0, 2, 1)
        N = x.shape[1]
        D1 = D_matrix(N)
        D2 = D_matrix(N-1)
        D = torch.mm(D2, D1).to(self.device)

        g = torch.matmul(torch.inverse(torch.eye(N).to(self.device) + self.lamb * torch.mm(D.T, D)), x)
        res = x - g
        return res, g
class Encoder(nn.Module):
    def __init__(self, in_dim):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(

            nn.Conv1d(in_dim, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128 , 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, 7, 1, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv1d(256, 512, 7, 1, 3),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, input):
        output = self.main(input)

        return output
class STEncoder(nn.Module):
    def __init__(self, input_dims,device):
        super().__init__()
        self.device=device
        self.input_dims = input_dims
        self.Decomp1 = hp_filter(lamb=6400,device=self.device)
        self.encoder = Encoder(in_dim = input_dims)

    def forward(self, x):  # x: batch x T x input_dims
        x = self.encoder(x.permute(0,2,1))   # encoder out : N x L x C(=d_model)
        res, trend = self.Decomp1(x)
        return trend, res ,x.permute(0,2,1)

