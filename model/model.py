
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.init as init

from layers import BlurPool
from blocks import DirectionalMultiScaleBlock, SGBlock
from attention import CHM


class TFSNet(nn.Module):
    def __init__(self, base_dim, mlp_ratio, drop_path_rate, num_classes=7):
        super().__init__()
        self.in_channel = base_dim

        self.stem = nn.Sequential(
            DirectionalMultiScaleBlock(in_channels=3, out_channels=self.in_channel)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]

        self.downsample = nn.Sequential(
            BlurPool(channels=self.in_channel, filt_size=3, stride=2),
            nn.Conv2d(self.in_channel, base_dim * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True)
        )
        in_ch = base_dim * 2

        self.blocks = nn.ModuleList([SGBlock(in_ch, mlp_ratio, dpr[i]) for i in range(4)])

        self.chm = CHM(in_ch)

        self.norm = nn.BatchNorm2d(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(in_ch, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.chm(x)

        x = self.norm(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)
