import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, dilation, groups, with_bn=True):
        super().__init__()
        self.add_module('conv',
                        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias=False))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BlurPool(nn.Module):
    def __init__(self, channels, filt_size=3, stride=2):
        super(BlurPool, self).__init__()
        self.stride = stride
        if filt_size == 3:
            a = torch.tensor([1., 2., 1.])

        k = a[:, None] * a[None, :]
        k = k / k.sum()
        kernel = k.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=1, groups=x.shape[1])
        out = out[:, :, ::self.stride, ::self.stride]
        return out