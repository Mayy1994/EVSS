import torch
import torch.nn.functional as F
import warnings

from torch import nn as nn


upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
batchnorm_momentum = 0.01 / 2


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x



class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1,
                 drop_rate=.0, separable=False):
        super(_BNReluConv, self).__init__()
        padding = k // 2
        conv_class = SeparableConv2d if separable else nn.Conv2d
        self.add_module('conv', conv_class(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias,
                                           dilation=dilation))
        
        if batch_norm:
            self.add_module('norm',  nn.BatchNorm2d(num_maps_out, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(drop_rate, inplace=True))


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True):
        super(_Upsample, self).__init__()
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        self.blend_conv2 = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv2.forward(x)

        return x


class SK_Block(nn.Module):
    def __init__(self, in_chan, mid_chan, *args, **kwargs):
        super(SK_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _BNReluConv(in_chan, mid_chan, k=1, batch_norm=True, separable=False)
        self.fc1 = nn.Linear(mid_chan, in_chan)
        self.fc2 = nn.Linear(mid_chan, in_chan)
        self.softmax = nn.Softmax(dim=1)
        self.conv_out = _BNReluConv(in_chan, in_chan, k=1, batch_norm=True, separable=False)



    def forward(self, feat1, feat2):
        feats = feat1 + feat2
        feat_s = self.avg_pool(feats)
        feat_z = self.fc(feat_s).squeeze()
        feat_1 = self.fc1(feat_z)
        feat_2 = self.fc2(feat_z)
        if len(feat_2.shape)== 1:
            feat_1 = feat_1.unsqueeze(0)
            feat_2 = feat_2.unsqueeze(0)

        feat_1_2 = torch.stack([feat_1, feat_2], 1).unsqueeze(-1)
        feat_1_2 = self.softmax(feat_1_2)
        feat_1 = feat_1_2[:,0,:,:].unsqueeze(-1)
        feat_2 = feat_1_2[:,1,:,:].unsqueeze(-1)
        feat1_new = feat1 * feat_1.expand_as(feat1)
        feat2_new = feat2 * feat_2.expand_as(feat2)
        feat_sum = self.conv_out(feat1_new + feat2_new)

        return feat_sum

class _Upsample_sk(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3, use_skip=True, only_skip=False,
                 detach_skip=False, fixed_size=None, separable=False, bneck_starts_with_bn=True):
        super(_Upsample_sk, self).__init__()
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn and bneck_starts_with_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.only_skip = only_skip
        self.detach_skip = detach_skip
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
        self.sk = SK_Block(num_maps_in, num_maps_in)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = self.sk(x, skip)
        x = self.blend_conv.forward(x)
        return x


class _UpsampleBlend(nn.Module):
    def __init__(self, num_features, use_bn=True, use_skip=True, detach_skip=False, fixed_size=None, k=3,
                 separable=False):
        super(_UpsampleBlend, self).__init__()
        self.blend_conv = _BNReluConv(num_features, num_features, k=k, batch_norm=use_bn, separable=separable)
        self.use_skip = use_skip
        self.detach_skip = detach_skip
        self.upsampling_method = upsample
        if fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)

    def forward(self, x, skip):
        if self.detach_skip:
            skip = skip.detach()
        skip_size = skip.size()[-2:]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.fixed_size = fixed_size
        self.grids = grids
        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))
        self.square_grid = square_grid
        self.upsampling_method = upsample
        if self.fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum,
                                                  batch_norm=use_bn and starts_with_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                            drop_rate=drop_rate))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1
        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = self.upsampling_method(level, target_size)
            levels.append(level)

        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x


class SpatialPyramidPooling_spp(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, drop_rate=.0,
                 fixed_size=None, starts_with_bn=True):
        super(SpatialPyramidPooling_spp, self).__init__()
        self.fixed_size = fixed_size
        self.grids = grids
        if self.fixed_size:
            ref = min(self.fixed_size)
            self.grids = list(filter(lambda x: x <= ref, self.grids))
        self.square_grid = square_grid
        self.upsampling_method = upsample
        if self.fixed_size is not None:
            self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum,
                                                  batch_norm=use_bn and starts_with_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn,
                                            drop_rate=drop_rate))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]

        ar = target_size[1] / target_size[0]

        levels.append(x)
        num = len(self.spp) - 1
        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = self.upsampling_method(level, target_size)
            levels.append(level)

        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
