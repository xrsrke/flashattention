
import torch
from flashattention.attention import Attention


def test_attention():
    query = torch.randn(3, 4, 10)
    key = torch.randn(3, 4, 10)
    value = torch.randn(3, 4, 10)
    mask = torch.zeros(3, 4, 4)
    mask[:, 0, 1] = 1
    mask[:, 1, 0] = 1
    mask[:, 2, 3] = 1
    mask[:, 3, 2] = 1
    attention = Attention(10, 10, 10)

    output, attn_weights = attention(query, key, value, mask)

    assert output.shape == (3, 4, 10)
    assert attn_weights.shape == (3, 4, 4)
