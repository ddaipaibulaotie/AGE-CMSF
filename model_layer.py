import torch
import torch.nn as nn
import math
from einops import rearrange
from timm.models.layers import trunc_normal_


class AGE(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(AGE, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            _x = torch.abs(x) if not self.after_relu else x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate

class AGELinear(nn.Linear):
    """
    兼容 nn.Linear 的子类：保留 .weight/.bias，支持原有 init_weights()
    输入: (B, L, C_in) 其中 C_in == in_features
    输出: (B, L, out_features)
    在 forward 中先用 AGE 做通道门控（把 L 当作伪空间维），再做线性映射。
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 age_mode: str = 'l2', age_after_relu: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.age = AGE(num_channels=in_features, mode=age_mode, after_relu=age_after_relu)

    def forward(self, entity_visual_tokens: torch.Tensor) -> torch.Tensor:
        # entity_visual_tokens: (B, L, C_in)
        B, L, C = entity_visual_tokens.shape
        assert C == self.in_features, f"Expected C_in={self.in_features}, got {C}"

        # (B, L, C) -> (B, C, L, 1) 作为 AGE 输入
        x = entity_visual_tokens.permute(0, 2, 1).unsqueeze(-1)
        x = self.age(x)                      # (B, C, L, 1)
        x = x.squeeze(-1).permute(0, 2, 1)   # (B, L, C)

        # 线性映射 (逐通道投影 C_in->out_features)，形状保持 (B, L, *)
        return super().forward(x)
