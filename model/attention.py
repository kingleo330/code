import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CHM(nn.Module):
    def __init__(self, channels, reduction):
        super(CHM, self).__init__()
        self.channels = channels
        self.query_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)

        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.channels ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return self.gamma * out + x