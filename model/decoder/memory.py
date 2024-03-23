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

        self.memory = None  # 存储级联表示的外部记忆

        # 定义注意力层
        self.query_proj = nn.Linear(cascade_dim, attn_dim)
        self.key_proj = nn.Linear(cascade_dim, attn_dim)
        self.value_proj = nn.Linear(cascade_dim, cascade_dim)

    def initialize_memory(self, cascade_repr):
        # 初始化外部记忆为一个全零张量
        self.memory = torch.zeros(self.memory_size, self.cascade_dim).to(self.device)
        self.memory[0] = cascade_repr.detach()  # 将当前级联表示存入记忆
        self.mem_ptr = 1  # 记忆指针,指向下一个可写入位置

    # def attend(self, cascade_repr):
    #     # 计算注意力权重
    #     query = self.query_proj(cascade_repr)  # (batch, attn_dim)
    #     keys = self.key_proj(self.memory[:self.mem_ptr])  # (mem_ptr, attn_dim)
    #     attn_weights = torch.bmm(query, keys.transpose(0, 1))  # (batch, mem_ptr)
    #     attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch, mem_ptr)
    #
    #     # 根据注意力权重聚合记忆
    #     values = self.value_proj(self.memory[:self.mem_ptr])  # (mem_ptr, cascade_dim)
    #     attended_repr = torch.bmm(attn_weights, values).squeeze(1)  # (batch, cascade_dim)
    #
    #     return attended_repr
    def attend(self, cascade_repr):
        # 计算注意力权重
        query = self.query_proj(cascade_repr)  # (batch, 1, attn_dim)
        keys = self.key_proj(self.memory[:self.mem_ptr])  # (mem_ptr, attn_dim)
        attn_weights = torch.matmul(query, keys.transpose(0, 1))  # (batch, 1, mem_ptr)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch, 1, mem_ptr)

        # 根据注意力权重聚合记忆
        values = self.value_proj(self.memory[:self.mem_ptr])  # (mem_ptr, cascade_dim)
        attended_repr = torch.matmul(attn_weights, values)  # (batch, cascade_dim)

        return attended_repr

    # def update_memory(self, cascade_repr):
    #     batch_size = cascade_repr.size(0)
    #     # 更新记忆
    #     if self.mem_ptr + batch_size <= self.memory_size:
    #         self.memory[self.mem_ptr:self.mem_ptr + batch_size] = cascade_repr
    #         self.mem_ptr += batch_size
    #     else:
    #
    #         self.memory = torch.cat([self.memory[self.mem_ptr + batch_size - self.memory_size:], cascade_repr], dim=0)
    #         self.mem_ptr = self.memory_size
    def update_memory(self, cascade_repr):
        batch_size = cascade_repr.size(0)
        # 分离级联表示的梯度信息
        cascade_repr_detached = cascade_repr.detach()

        # 更新记忆
        if self.mem_ptr + batch_size <= self.memory_size:
            self.memory[self.mem_ptr:self.mem_ptr + batch_size] = cascade_repr_detached
            self.mem_ptr += batch_size
        else:
            self.memory = torch.cat([self.memory[self.mem_ptr + batch_size - self.memory_size:], cascade_repr_detached],
                                    dim=0)
            self.mem_ptr = self.memory_size

    def forward(self, cascade_repr):
        if self.memory is None:
            self.initialize_memory(cascade_repr)
            return cascade_repr
        else:
            attended_repr = self.attend(cascade_repr)
            # self.update_memory(cascade_repr)
            return attended_repr
