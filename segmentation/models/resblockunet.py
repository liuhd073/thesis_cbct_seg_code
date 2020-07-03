# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


def is_power2(num):
    """Check if is power of 2"""
    return num != 0 and ((num & (num - 1)) == 0)


def get_activation(activation_name, **kwargs):
    if activation_name == 'relu':
        activation = F.relu
    elif activation_name == 'elu':
        activation = F.elu
    elif activation_name == 'lrelu':
        negative_slope = kwargs.get('activation_negative_slope', 0.01)
        activation = lambda input: F.leaky_relu(input, negative_slope)
    else:
        raise NotImplementedError(f'{activation_name} not available.')

    return activation


class ResBlockUnet(nn.Module):
    def __init__(self, num_channels, num_classes, output_shape,
                 down_blocks=((3, 0, 32), (3, 0, 32), (3, 0, 64), (1, 2, 64), (1, 2, 128), (1, 2, 128), (0, 4, 256, False)),
                 up_blocks=((4, 256, False), (4, 128), (4, 128), (3, 64), (3, 64), (3, 64), (3, 64)),
                 bottleneck_channels=1024, mode='bilinear', dim=3, compress_dim=0, dropout_prob=0.0,
                 use_group_norm=False, num_groups=4, activation='relu'):
        """
        DeepMind U-Net model implementation
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        assert len(down_blocks) == len(up_blocks), 'Upsampling path should have the same length as downsampling.'
        assert len(output_shape) == dim, f'Output shape should have {dim} dimensions.'
        self.depth = len(down_blocks)
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.shape_out = output_shape
        self.shape_in = self.compute_shape_in(output_shape, compress_dim=compress_dim)
        self.dim = dim
        self.compress_dim = compress_dim
        # If our model output has only one slice in compressed dim, we use 2D convolutions in in the upsampling path
        if dim == 3 and output_shape[compress_dim] == 1:
            self.dim_up = 2
        else:
            self.dim_up = self.dim
        self.shape_bottleneck = self.compute_shape_bottleneck_from_output(output_shape, compress_dim=compress_dim)
        self.logger.info(f'Input shape {self.shape_in}, output shape {self.shape_out}, bottleneck shape {self.shape_bottleneck}')
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        curr_channels = num_channels
        for desc_down in down_blocks:
            if len(desc_down) == 3:
                downsample = True
            else:
                downsample = desc_down[3]
            down_block = ResBlock(curr_channels, desc_down[2], n_conv_a=desc_down[0], n_conv_b=desc_down[1], dim=dim,
                              compress_dim=compress_dim, dropout_prob=dropout_prob, use_group_norm=use_group_norm,
                              num_groups=num_groups, activation=activation, downsample=downsample)
            self.downs.append(down_block)
            curr_channels = desc_down[2]

        self.bottleneck = BottleneckBlock(self.shape_bottleneck, curr_channels, int_ch=bottleneck_channels)

        for idx, desc_up in enumerate(up_blocks):
            if len(desc_up) == 2:
                upsample = True
            else:
                upsample = desc_up[2]
            up_block = UpsampleBlock(curr_channels + down_blocks[self.depth - idx - 1][2], desc_up[1], desc_up[0], dim=dim,
                                     conv_dim=self.dim_up, compress_dim=compress_dim, dropout_prob=dropout_prob,
                                     use_group_norm=use_group_norm, num_groups=num_groups, upscale=upsample, mode=mode,
                                     activation=activation)
            curr_channels = desc_up[1]
            self.ups.append(up_block)

        if self.dim_up == 2:
            self.outc = nn.Conv2d(curr_channels, num_classes, 1, padding=0)
        else:
            self.outc = nn.Conv3d(curr_channels, num_classes, 1, padding=0)

        #self.init_weights()

    def forward(self, x):
        xs = []
        for idx in range(self.depth):
            y, x = self.downs[idx](x)
            xs.append(y)

        x = self.bottleneck(xs[-1])
        if self.dim_up < self.dim:
            x = torch.squeeze(x, dim=(2 + self.compress_dim))
        for idx in range(self.depth):
            x = self.ups[idx](x, xs[self.depth - idx - 1])

        x = self.outc(x)
        if self.dim_up < self.dim:
            # Input has a compressed dimension which we restore to have a correct output shape
            x = torch.unsqueeze(x, 2 + self.compress_dim)

        return x

    def compute_shape_bottleneck_from_output(self, shape_out, compress_dim=1):
        tshape = list(shape_out)
        dims = [val for val in range(len(shape_out)) if val != compress_dim]
        for block in self.up_blocks:
            if len(block) == 2:
                upsample = True
            else:
                upsample = block[2]
            if upsample:
                for dim in dims:
                    tshape[dim] = tshape[dim] // 2
        return tshape

    def compute_shape_in(self, shape_out, compress_dim=1):
        tshape = list(shape_out)
        dims = [val for val in range(len(shape_out)) if val != compress_dim]

        assert np.all([is_power2(tshape[dim]) for dim in dims]),\
            'Output dimensions in the non-compressed dimension should be powers of two'

        for block in self.up_blocks:
            if len(block) == 2:
                upsample = True
            else:
                upsample = block[2]
            if upsample:
                for dim in dims:
                    tshape[dim] = tshape[dim] // 2
        for block in self.down_blocks:
            if len(block) == 3:
                downsample = True
            else:
                downsample = block[3]
            if downsample:
                for dim in dims:
                    tshape[dim] = tshape[dim] * 2
            tshape[compress_dim] = tshape[compress_dim] + 2*block[1]
        return tshape

    def compute_shape_out(self, shape_in, compress_dim=1):
        tshape = list(shape_in)
        dims = [val for val in range(len(shape_in)) if val != compress_dim]

        assert np.all([is_power2(tshape[dim]) for dim in dims]), \
            'Output dimensions in the non-compressed dimension should be powers of two'

        for block in self.down_blocks:
            if len(block) == 3:
                downsample = True
            else:
                downsample = block[3]
            if downsample:
                for dim in dims:
                    tshape[dim] = tshape[dim] // 2
            tshape[compress_dim] = tshape[compress_dim] - 2 * block[1]
        for block in self.up_blocks:
            if len(block) == 2:
                upsample = True
            else:
                upsample = block[2]
            if upsample:
                for dim in dims:
                    tshape[dim] = tshape[dim] * 2
        return tshape


class BottleneckBlock(nn.Module):
    def __init__(self, shape_in, in_ch, int_ch=1024, activation='relu'):
        super(BottleneckBlock, self).__init__()
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise NotImplementedError("{} not available.".format(activation))
        input_size = np.prod(list(shape_in))
        self.fc1 = nn.Linear(input_size*in_ch, int_ch)
        self.fc2 = nn.Linear(int_ch, int_ch)
        self.fc3 = nn.Linear(int_ch, int_ch)
        self.fc4 = nn.Linear(int_ch, input_size*in_ch)

    def forward(self, x):
        in_shape = list(x.size())
        y = self.fc1(x.view(in_shape[0], -1))

        z = self.fc2(self.activation(y, inplace=True)) + y
        w = self.fc3(self.activation(z, inplace=True)) + z
        w = self.fc4(self.activation(w, inplace=True))

        return w.view(*in_shape)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv, dim=3, conv_dim=3, compress_dim=1, dropout_prob=0.0, use_group_norm=False,
                 num_groups=4, upscale=True, mode='bilinear', activation='relu'):
        super(UpsampleBlock, self).__init__()

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise NotImplementedError("{} not available.".format(activation))
        self.dropout_prob = dropout_prob
        self.dim = dim
        self.conv_dim = conv_dim
        self.compress_dim = compress_dim
        self.upscale = upscale
        if conv_dim == 2:
            us_ker = 2
            kernel_size = 3
        else:
            us_ker = [2, 2, 2]
            us_ker[compress_dim] = 1
            kernel_size = [3, 3, 3]
            kernel_size[compress_dim] = 1
        if upscale:
            if mode in ['bilinear', 'nearest']:
                self.up = lambda x_in: torch.nn.functional.interpolate(
                    x_in, mode=mode, scale_factor=us_ker, align_corners=False) #, recompute_scale_factor=True)
            else:
                raise NotImplementedError("Mode {} not available.".format(mode))

        self.convs = nn.ModuleList()

        if conv_dim == 2:
            self.adaptation = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.adaptation = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        t = in_ch
        if use_group_norm:
            for _ in range(n_conv):
                if conv_dim == 3:
                    self.convs.append(nn.Sequential(nn.Conv3d(t, out_ch, kernel_size=kernel_size, padding=1),
                                                   nn.GroupNorm(num_groups, out_ch)))
                else:
                    self.convs.append(nn.Sequential(nn.Conv2d(t, out_ch, kernel_size=kernel_size, padding=1),
                                                   nn.GroupNorm(num_groups, out_ch)))
                t = out_ch
        else:
            for _ in range(n_conv):
                if conv_dim == 3:
                    self.convs.append(nn.Conv3d(t, out_ch, kernel_size=kernel_size, padding=1))
                else:
                    self.convs.append(nn.Conv2d(t, out_ch, kernel_size=kernel_size, padding=1))
                t = out_ch


    def forward(self, x2, x1):
        # x2 for upsample input
        # x1 for skip connections
        x1_shape = list(x1.size())
        x2_shape = list(x2.size())
        if self.dim == 3:
            if self.conv_dim == 3:
                # Crop tensor from skip-connections to 3d patch in compressed axis
                delta = (x1_shape[2 + self.compress_dim] - x2_shape[2 + self.compress_dim]) // 2
                if self.compress_dim == 0:
                    x = torch.cat([x1[:, :, delta: x1_shape[2] - delta, :, :], self.up(x2) if self.upscale else x2], dim=1)
                elif self.compress_dim == 1:
                    x = torch.cat([x1[:, :, :, delta: x1_shape[3] - delta, :], self.up(x2) if self.upscale else x2], dim=1)
                else:
                    x = torch.cat([x1[:, :, :, :, delta: x1_shape[4] - delta], self.up(x2) if self.upscale else x2], dim=1)
            else:
                # Crop skip-connections to patch pf two-dimensional features.
                if self.compress_dim == 0:
                    x = torch.cat([x1[:, :, x1_shape[2] // 2, :, :], self.up(x2) if self.upscale else x2], dim=1)
                elif self.compress_dim == 1:
                    x = torch.cat([x1[:, :, :, x1_shape[3] // 2, :], self.up(x2) if self.upscale else x2], dim=1)
                else:
                    x = torch.cat([x1[:, :, :, :, x1_shape[4] // 2], self.up(x2) if self.upscale else x2], dim=1)
        else:
            x = torch.cat([x1, self.up(x2)], dim=1)

        y = self.activation(x, inplace=True)
        for conv in self.convs:
            y = conv(y)
            y = self.activation(y, inplace=True)

        x = self.adaptation(x)
        return x + y


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv_a=1, n_conv_b=1, dim=3, compress_dim=1, dropout_prob=0.0, use_group_norm=False,
                 num_groups=4, downsample=True, activation='relu'):
        super(ResBlock, self).__init__()
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise NotImplementedError("{} not available.".format(activation))
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.dim = dim
        self.compress_dim = compress_dim
        self.n_conv_a = n_conv_a
        self.n_conv_b = n_conv_b
        self.dropout_prob = dropout_prob
        self.downsample = downsample
        if dim == 3:
            kernel_size = [3, 3, 3]
            kernel_size[compress_dim] = 1
            kernel_size_cmp = [1, 1, 1]
            kernel_size_cmp[compress_dim] = 3
            padding_cmp = [1, 1, 1]
            padding_cmp[compress_dim] = 0
        else:
            padding_cmp = 1
            kernel_size = 3
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.adaptation = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        t = in_ch
        if not use_group_norm:
            for _ in range(n_conv_a):
                self.conv_a.append(conv(t, out_ch, kernel_size=kernel_size, padding=padding_cmp))
                t = out_ch
            for _ in range(n_conv_b):
                if dim == 3:
                    self.conv_b.append(nn.Sequential(nn.Conv3d(t, out_ch, kernel_size=kernel_size, padding=padding_cmp),
                                                      nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size_cmp, padding=0)))
                    t = out_ch
                else:
                    self.conv_b.append(nn.Conv2d(t, out_ch, kernel_size=kernel_size, padding=1))
        else:
            for _ in range(n_conv_a):
                self.conv_a.append(nn.Sequential(conv(t, out_ch, kernel_size=kernel_size, padding=padding_cmp),
                                                 nn.GroupNorm(num_groups, out_ch)))
                t = out_ch
            for _ in range(n_conv_b):
                if dim == 3:
                    self.conv_b.append(nn.Sequential(nn.Conv3d(t, out_ch, kernel_size=kernel_size, padding=padding_cmp),
                                                     nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size_cmp, padding=0),
                                                     nn.GroupNorm(num_groups, out_ch)))
                    t = out_ch
                else:
                    self.conv_b.append(nn.Sequential(nn.Conv2d(t, out_ch, kernel_size=kernel_size, padding=1),
                                                     nn.GroupNorm(num_groups, out_ch)))

        if dim == 2:
            self.pooling = nn.AvgPool2d(2)
        else:
            ks = [2, 2, 2]
            ks[compress_dim] = 1
            self.pooling = nn.AvgPool3d(kernel_size=ks)
            # self.pooling = nn.MaxPool3d(kernel_size=ks)
            # self.pooling = nn.Conv3d(out_ch, out_ch, ks, stride=ks)

    def forward(self, x):
        in_shape = list(x.size())

        y = self.activation(x, inplace=True)
        for conv in self.conv_a:
            y = conv(y)
            y = self.activation(y, inplace=True)
            if self.dropout_prob and self.dropout_prob > 0:
                y = F.dropout(y, p=self.dropout_prob, training=self.training)
        for i, conv in enumerate(self.conv_b):
            y = conv(y)
            if i < self.n_conv_b - 1:
                y = self.activation(y, inplace=True)
            if self.dropout_prob and self.dropout_prob > 0:
                y = F.dropout(y, p=self.dropout_prob, training=self.training)
        if self.dim == 3:
            if self.compress_dim == 0:
                y = y + self.adaptation(x[:, :, self.n_conv_b: in_shape[2] - self.n_conv_b, :, :])
            elif self.compress_dim == 1:
                y = y + self.adaptation(x[:, :, :, self.n_conv_b: in_shape[3] - self.n_conv_b, :])
            else:
                y = y + self.adaptation(x[:, :, :, :, self.n_conv_b: in_shape[4] - self.n_conv_b])
        else:
            y = y + self.adaptation(x)  # No cropping needed for 2D models.

        if self.downsample:
            return y, self.pooling(y)
        else:
            return y, y
