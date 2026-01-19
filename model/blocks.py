import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from layers import ConvBN, SEBlock


class DirectionalMultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DirectionalMultiScaleBlock, self).__init__()
        self.freq_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 1), padding=(3, 0))
        self.time_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 7), padding=(0, 3))
        self.cross_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels * 3)
        self.output_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f = self.freq_conv(x)
        t = self.time_conv(x)
        c = self.cross_conv(x)
        out = torch.cat([f, t, c], dim=1)
        out = self.se(out)
        out = self.output_conv(out)
        return self.relu(out)


class GateModule(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        return self.fc(w)


class SAFM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.AvgPool2d(4)
        self.conv = nn.Conv2d(in_channels * 3, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x1 = F.interpolate(x1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        feat = torch.cat([x, x1, x2], dim=1)
        attn = self.sigmoid(self.conv(feat))
        return x * attn


class SGBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = nn.ReLU6()

        self.gate = GateModule(dim)
        self.safm = SAFM(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))

        gate = self.gate(identity)
        fused = gate * x + (1 - gate) * identity
        fused = fused + self.drop_path(fused)

        out = self.safm(fused)
        return out