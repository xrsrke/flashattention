import torch
from torch import nn
import torch.nn.functional as F


# write regular scale-dot product attention
class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout=0.1):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout
        self.scale = 1 / (key_dim ** 0.5)
        self.query = nn.Linear(query_dim, key_dim)
        self.key = nn.Linear(key_dim, key_dim)
        self.value = nn.Linear(value_dim, value_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
