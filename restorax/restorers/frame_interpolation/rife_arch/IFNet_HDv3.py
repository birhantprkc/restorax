# Vendored from hzwer/Practical-RIFE (MIT License)
# https://github.com/hzwer/Practical-RIFE/blob/main/model/IFNet_HDv3.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x1 = self.relu(self.cnn1(x0))
        x2 = self.relu(self.cnn2(x1))
        x3 = self.cnn3(x2)
        if feat:
            return x3, x0
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        # block0: cat(img0(3), img1(3), f0(8), f1(8)) = 22ch, no flow yet
        # block1+: same 22ch input + flow(4ch) appended inside IFBlock.forward = 26ch
        self.block0 = IFBlock(22, c=192)
        self.block1 = IFBlock(26, c=128)
        self.block2 = IFBlock(26, c=96)
        self.block3 = IFBlock(26, c=64)
        self.encode = Head()

    def forward(self, x, timestep=0.5, scale_list=None, training=False, fastmode=True, ensemble=False):
        if scale_list is None:
            scale_list = [8, 4, 2, 1]

        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        f0, x0 = self.encode(img0, feat=True)
        f1, x1 = self.encode(img1, feat=True)

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        block = [self.block0, self.block1, self.block2, self.block3]
        for i, blk in enumerate(block):
            if flow is None:
                flow, mask = blk(torch.cat((img0, img1, f0, f1), 1), None, scale=scale_list[i])
            else:
                wf0 = self._warp(f0, flow[:, :2])
                wf1 = self._warp(f1, flow[:, 2:4])
                fd, md = blk(torch.cat((warped_img0, warped_img1, wf0, wf1), 1), flow, scale=scale_list[i])
                mask = md
                flow = flow + fd

            flow_list.append(flow)
            mask_list.append(torch.sigmoid(mask))

            warped_img0 = self._warp(img0, flow[:, :2])
            warped_img1 = self._warp(img1, flow[:, 2:4])

            if not fastmode or i == 3:
                merged.append(
                    warped_img0 * mask_list[-1] + warped_img1 * (1 - mask_list[-1])
                )

        return merged[-1], flow_list, mask_list

    @staticmethod
    def _warp(x, flow):
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)  # (1, 2, H, W)
        vgrid = grid + flow
        vgrid_x = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        return F.grid_sample(x, vgrid_scaled, mode="bilinear", padding_mode="border", align_corners=True)
