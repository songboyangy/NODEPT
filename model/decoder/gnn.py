import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
#import lib.utils as utils
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_add


class CasSelf(nn.Module):
    def __init__(self, in_features, out_features,dropout=0.2):
        super(CasSelf, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.evolve=nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, out_features))
        self.dropout = nn.Dropout(p=dropout)
        # Define layers or parameters here

    def forward(self, x):
        x=x+self.evolve(x)
        x=self.dropout(x)
        # Define the forward pass
        return x
