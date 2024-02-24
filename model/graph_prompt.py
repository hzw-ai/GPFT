import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot

from torch_geometric.nn import GATConv


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int, dtype, **kwargs):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels).to(dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int = None, dtype=torch.float64):
        super(GPFplusAtt, self).__init__()
        if not p_num:
            p_num = 2
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels).to(dtype=dtype))
        self.a = nn.Linear(in_channels, p_num, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=-1)
        p = weight.matmul(self.p_list)

        return x + p
