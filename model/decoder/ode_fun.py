import torch
import torch.nn as nn
from model.decoder.memory import ExternalMemory


class CasSelf(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(CasSelf, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.evolve = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, out_features))
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        x = x + self.evolve(x)
        x = self.dropout(x)
        # Define the forward pass
        return x


class CasExternalMemory(nn.Module):
    def __init__(self, in_features, out_features, external_memory: ExternalMemory, hidden_dim=64, num_layers=2,
                 dropout=0.2):
        super(CasExternalMemory, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.external_memory = external_memory
        self.dropout=nn.Dropout(p=dropout)
        self.lin=nn.Linear(in_features + hidden_dim, out_features)

        # 构建多层级联结构
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features + hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features + hidden_dim, out_features))

        self.evolve = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        external_memory_repr = self.external_memory(x)
        x = torch.cat([x, external_memory_repr], dim=1)
        x=self.lin(x)
        x=self.dropout(x)

        return x
