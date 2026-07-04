# Vendored from https://github.com/DachunKai/EvTexture (Apache License 2.0)
# Sources: basicsr/archs/evtexture_arch.py, spynet_arch.py, unet_arch.py, arch_util.py
# (SpyNet/UNet/shared ops inlined here to avoid a hard `basicsr` runtime dependency;
# imports adapted for restorax, ARCH_REGISTRY decorators dropped)
# See https://github.com/DachunKai/EvTexture/blob/main/LICENSE for full terms.
import math

import torch
from torch import nn as nn
from torch.nn import functional as F


# ---- from basicsr/archs/arch_util.py ----

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN."""

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow."""
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False

    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    return output


def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)


class SizeAdapter(object):
    """Pads/unpads inputs to a multiple of `minimum_size` so UNet can process arbitrary sizes."""

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]


class ConvGRU(nn.Module):

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR."""

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


class SmallUpdateBlock(nn.Module):

    def __init__(self, hidden_dim=64, input_dim=64 * 2):
        super(SmallUpdateBlock, self).__init__()
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.res_head = ConvResidualBlocks(num_in_ch=64, num_out_ch=64, num_block=5)

    def forward(self, net, context, motion):
        inp = torch.cat([context, motion], dim=1)
        net = self.gru(net, inp)
        delta_net = self.res_head(net)
        return net, delta_net


# ---- from basicsr/archs/spynet_arch.py ----

class BasicModule(nn.Module):
    """Basic Module for SpyNet."""

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture."""

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            self.load_state_dict(
                torch.load(load_path, map_location=lambda storage, loc: storage, weights_only=True)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow = F.interpolate(input=self.process(ref, supp), size=(h, w), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow


# ---- from basicsr/archs/unet_arch.py ----
# Modified from timelens: https://github.com/uzh-rpg/rpg_timelens/blob/main/timelens/superslomo/unet.py

class up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class UNet(nn.Module):
    """Modified version of Unet from SuperSloMo.

    Difference:
    1) there is an option to skip ReLU after the last convolution.
    2) there is a size adapter module that makes sure that input of all sizes
       can be processed correctly. It is necessary because original
       UNet can process only inputs with spatial dimensions divisible by 32.
    """

    def __init__(self, inChannels, outChannels, ends_with_relu=True, load_path=None):
        super(UNet, self).__init__()
        self._ends_with_relu = ends_with_relu
        self._size_adapter = SizeAdapter(minimum_size=32)

        # 5-level
        self.conv1 = nn.Conv2d(inChannels, 8, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.down1 = down(8, 16, 5)
        self.down2 = down(16, 32, 3)
        self.down3 = down(32, 64, 3)
        self.down4 = down(64, 128, 3)
        self.down5 = down(128, 128, 3)
        self.up1 = up(128, 128)
        self.up2 = up(128, 64)
        self.up3 = up(64, 32)
        self.up4 = up(32, 16)
        self.up5 = up(16, 8)
        self.conv3 = nn.Conv2d(8, outChannels, 3, stride=1, padding=1)

        if load_path:
            self.load_state_dict(
                torch.load(load_path, map_location=lambda storage, loc: storage, weights_only=True)['params_ema'])

    def forward(self, x):
        x = self._size_adapter.pad(x)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)

        if self._ends_with_relu:
            x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        else:
            x = self.conv3(x)
        x = self._size_adapter.unpad(x)
        return x


# ---- from basicsr/archs/evtexture_arch.py ----

class EvTexture(nn.Module):
    """EvTexture: Event-driven Texture Enhancement for Video Super-Resolution (ICML 2024)
       Note that: this class is for 4x VSR

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 30
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
        scale (int): accepted for call-site compatibility with restorax's
            EvTextureRestorer (`EvTexture(scale=4)`); upstream has no such
            argument since this architecture is fixed at 4x. Unused beyond
            an assertion.
            ponytail: kept as a no-op kwarg instead of touching evtexture.py.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None, scale=4):
        super().__init__()
        assert scale == 4, "EvTexture only supports 4x super-resolution"
        self.num_feat = num_feat

        # RGB-based flow alignment
        self.spynet = SpyNet(spynet_path)
        self.cnet = ConvResidualBlocks(num_in_ch=3, num_out_ch=64, num_block=8)

        # iterative texture enhancement module
        self.enet = UNet(inChannels=1, outChannels=num_feat)
        self.update_block = SmallUpdateBlock(hidden_dim=num_feat, input_dim=num_feat * 2)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat * 2 + 3, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    # context feature extractor
    def get_feat(self, x):
        b, n, c, h, w = x.size()
        feats_ = self.cnet(x.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(b, n, -1, h, w)

        return feats_

    def forward(self, imgs, events):
        """Forward function of EvTexture.

        Args:
            imgs: Input frames with shape (b, n, c, h, w).
            events: Simulated event polarity maps with shape (b, n, 2, h, w),
                as produced by EvTextureRestorer._simulate_events.

        Output:
            out_l: output frames with shape (b, n, c, 4h, 4w)

        ponytail: upstream expects independently-captured forward/backward
        event voxel grids (voxels_f, voxels_b), each of length n-1. The
        restorer only simulates one symmetric polarity map per frame (no
        real event camera), so both directions reuse the same n-1-length
        slice here. Wire up true directional voxels if real event capture
        is ever added.
        """
        voxels_f = events[:, 1:]
        voxels_b = events[:, 1:]

        flows_forward, flows_backward = self.get_flow(imgs)
        feat_imgs = self.get_feat(imgs)
        b, n, _, h, w = imgs.size()
        bins = voxels_f.size()[2]

        # backward branch
        out_l = []
        feat_prop = imgs.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = imgs[:, i, :, :, :]

            if i < n - 1:
                # motion branch by rgb frames
                flow = flows_backward[:, i, :, :, :]
                feat_prop_coarse = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                # texture branch by event voxels
                hidden_state = feat_prop.clone()
                feat_img = feat_imgs[:, i, :, :, :]  # [B, num_feat, H, W]
                cur_voxel = voxels_f[:, i, :, :, :]  # [B, Bins, H, W]

                # iterative update block
                feat_prop_fine = feat_prop.clone()
                for j in range(bins - 1, -1, -1):
                    voxel_j = cur_voxel[:, j, :, :].unsqueeze(1)  # [B, 1, H, W]
                    feat_motion = self.enet(voxel_j)  # [B, num_feat, H, W]
                    hidden_state, delta_feat = self.update_block(hidden_state, feat_img, feat_motion)
                    feat_prop_fine = feat_prop_fine + delta_feat

                feat_prop = self.fusion(torch.cat([feat_prop_fine, feat_prop_coarse], dim=1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = imgs[:, i, :, :, :]

            if i > 0:
                # motion branch by rgb frames
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop_coarse = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                # texture branch by event voxels
                hidden_state = feat_prop.clone()
                feat_img = feat_imgs[:, i, :, :, :]  # [B, num_feat, H, W]
                cur_voxel = voxels_b[:, i - 1, :, :, :]  # [B, Bins, H, W]

                # iterative update block
                feat_prop_fine = feat_prop.clone()
                for j in range(bins - 1, -1, -1):
                    voxel_j = cur_voxel[:, j, :, :].unsqueeze(1)  # [B, 1, H, W]
                    feat_motion = self.enet(voxel_j)  # [B, num_feat, H, W]
                    hidden_state, delta_feat = self.update_block(hidden_state, feat_img, feat_motion)
                    feat_prop_fine = feat_prop_fine + delta_feat

                feat_prop = self.fusion(torch.cat([feat_prop_fine, feat_prop_coarse], dim=1))

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)
