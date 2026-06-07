"""YOLOP-style multi-task model.

This is a self-contained recreation of the paper architecture:
- one shared RGB backbone
- one anchor-based detection head with 3 scales
- one drivable-area segmentation head
- one lane-line segmentation head

The implementation mirrors the original building blocks used in YOLOP
(`Focus`, `Conv`, `Bottleneck`, `BottleneckCSP`, `SPP`, `Concat`, `Detect`).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k: int, p: int | None = None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3.0, 0.0, 6.0) / 6.0


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    stride = None

    def __init__(self, nc=13, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors, dtype=torch.float32).view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def check_anchor_order(m: Detect):
    anchor_area = m.anchors.prod(-1).mean(-1)
    if anchor_area.numel() > 1 and anchor_area[-1] < anchor_area[0]:
        m.anchors = torch.flip(m.anchors, [0])
        m.anchor_grid = torch.flip(m.anchor_grid, [0])


def initialize_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


class SegHead(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int = 2):
        super().__init__()
        self.reduce = Conv(in_channels, 64, 3, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse1 = BottleneckCSP(64 + skip_channels, 64, 1, False)
        self.conv2 = Conv(64, 32, 3, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = Conv(32, 16, 3, 1)
        self.fuse2 = BottleneckCSP(16, 8, 1, False)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.out = Conv(8, out_channels, 3, 1)

    def forward(self, x, skip):
        x = self.reduce(x)
        x = self.up1(x)
        x = self.fuse1(torch.cat((x, skip), 1))
        x = self.conv2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.fuse2(x)
        x = self.up3(x)
        return self.out(x)


DEFAULT_ANCHORS = [
    [3, 9, 5, 11, 4, 20],
    [7, 18, 6, 39, 12, 31],
    [19, 50, 38, 81, 68, 157],
]


class YOLOP(nn.Module):
    def __init__(self, nc: int = 13, anchors: Sequence[Sequence[int]] = DEFAULT_ANCHORS):
        super().__init__()
        self.nc = nc
        self.names = [str(i) for i in range(nc)]

        # Backbone
        self.focus = Focus(3, 32, 3)
        self.conv1 = Conv(32, 64, 3, 2)
        self.csp2 = BottleneckCSP(64, 64, 1)
        self.conv3 = Conv(64, 128, 3, 2)
        self.csp4 = BottleneckCSP(128, 128, 3)
        self.conv5 = Conv(128, 256, 3, 2)
        self.csp6 = BottleneckCSP(256, 256, 3)
        self.conv7 = Conv(256, 512, 3, 2)
        self.spp8 = SPP(512, 512, [5, 9, 13])
        self.csp9 = BottleneckCSP(512, 512, 1, False)

        # FPN/PAN neck for detection
        self.cv10 = Conv(512, 256, 1, 1)
        self.up11 = nn.Upsample(scale_factor=2, mode='nearest')
        self.csp13 = BottleneckCSP(512, 256, 1, False)
        self.cv14 = Conv(256, 128, 1, 1)
        self.up15 = nn.Upsample(scale_factor=2, mode='nearest')
        self.csp17 = BottleneckCSP(256, 128, 1, False)
        self.cv18 = Conv(128, 128, 3, 2)
        self.csp20 = BottleneckCSP(256, 256, 1, False)
        self.cv21 = Conv(256, 256, 3, 2)
        self.csp23 = BottleneckCSP(512, 512, 1, False)
        self.detect = Detect(nc=nc, anchors=anchors, ch=(128, 256, 512))

        # Segmentation heads
        self.da_head = SegHead(128, 64, 2)
        self.ll_head = SegHead(128, 64, 2)

        initialize_weights(self)

        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        check_anchor_order(self.detect)
        self._initialize_biases()

    def _initialize_biases(self, cf=None):
        m = self.detect
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        x0 = self.focus(x)
        x1 = self.conv1(x0)
        x2 = self.csp2(x1)
        x3 = self.conv3(x2)
        x4 = self.csp4(x3)
        x5 = self.conv5(x4)
        x6 = self.csp6(x5)
        x7 = self.conv7(x6)
        x8 = self.spp8(x7)
        x9 = self.csp9(x8)

        det_in_0 = self.cv10(x9)
        det_in_0_up = self.up11(det_in_0)
        det_in_1 = self.csp13(torch.cat((det_in_0_up, x6), 1))
        det_in_1_red = self.cv14(det_in_1)
        det_in_1_up = self.up15(det_in_1_red)
        det_p3 = self.csp17(torch.cat((det_in_1_up, x4), 1))
        det_p4 = self.csp20(torch.cat((self.cv18(det_p3), det_in_1_red), 1))
        det_p5 = self.csp23(torch.cat((self.cv21(det_p4), det_in_0), 1))

        det_out = self.detect([det_p3, det_p4, det_p5])

        da_seg = torch.sigmoid(self.da_head(det_p3, x2))
        ll_seg = torch.sigmoid(self.ll_head(det_p3, x2))

        return det_out, da_seg, ll_seg


def get_net(cfg=None, **kwargs):
    return YOLOP(**kwargs)


if __name__ == '__main__':
    model = get_net()
    model.eval()
    sample = torch.randn(1, 3, 640, 640)
    det_out, da_seg, ll_seg = model(sample)
    print('det_out type:', type(det_out))
    if isinstance(det_out, tuple):
        print('det predictions:', det_out[0].shape)
        print('det raw scales:', [x.shape for x in det_out[1]])
    else:
        print('det raw scales:', [x.shape for x in det_out])
    print('da_seg:', da_seg.shape)
    print('ll_seg:', ll_seg.shape)