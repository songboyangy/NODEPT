import numpy as np
import torch
import torch.nn as nn


class ExternalMemory(nn.Module):
    def __init__(self, cascade_dim, device, memory_size=50, attn_dim=128):
        super(ExternalMemory, self).__init__()
        self.device = device
        self.mem_ptr = 0
        self.memory_size = memory_size
        self.cascade_dim = cascade_dim
        self.attn_dim = attn_dim

        self.memory = None


        self.query_proj = nn.Linear(cascade_dim, attn_dim)
        self.key_proj = nn.Linear(cascade_dim, attn_dim)
        self.value_proj = nn.Linear(cascade_dim, cascade_dim)

    def initialize_memory(self, cascade_repr):

        self.memory = torch.zeros(self.memory_size, self.cascade_dim).to(self.device)
        batch_size = cascade_repr.size(0)

        cascade_repr_detached = cascade_repr.detach()
        self.memory[self.mem_ptr:self.mem_ptr + batch_size] = cascade_repr_detached
        self.mem_ptr += batch_size

    def attend(self, cascade_repr):

        query = self.query_proj(cascade_repr)  # (batch, 1, attn_dim)
        keys = self.key_proj(self.memory[:self.mem_ptr])  # (mem_ptr, attn_dim)
        attn_weights = torch.matmul(query, keys.transpose(0, 1))  # (batch, 1, mem_ptr)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch, 1, mem_ptr)


        values = self.value_proj(self.memory[:self.mem_ptr])  # (mem_ptr, cascade_dim)
        attended_repr = torch.matmul(attn_weights, values)  # (batch, cascade_dim)

        return attended_repr

    def update_memory(self, cascade_repr):
        batch_size = cascade_repr.size(0)

        cascade_repr_detached = cascade_repr.detach()


        if self.mem_ptr + batch_size <= self.memory_size:
            self.memory[self.mem_ptr:self.mem_ptr + batch_size] = cascade_repr_detached
            self.mem_ptr += batch_size
        else:
            self.memory = torch.cat([self.memory[self.mem_ptr + batch_size - self.memory_size:], cascade_repr_detached],
                                    dim=0)
            self.mem_ptr = self.memory_size

    def reset_memory(self):

        self.memory = None
        self.mem_ptr = 0

    def forward(self, cascade_repr):
        if self.memory is None:
            self.initialize_memory(cascade_repr)
            return cascade_repr
        else:
            attended_repr = self.attend(cascade_repr)
            # self.update_memory(cascade_repr)
            return attended_repr
