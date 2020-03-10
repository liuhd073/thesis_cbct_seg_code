'''
Author: Tessa Wagenaar
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from resblockunet import ResBlock, UpsampleBlock


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
                    x_in, mode=mode, scale_factor=us_ker, align_corners=False)
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
            # self.pooling = nn.AvgPool3d(kernel_size=ks)
            self.pooling = nn.Conv3d(out_ch, out_ch, ks, stride=ks)

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


class UNetResBlocks(nn.Module):
    def __init__(self):
        super(UNetResBlocks, self).__init__()
        self.res1 = ResBlock(1, 32, 3, 0, dim=3, compress_dim=0, downsample=True)
        # self.res2 = ResBlock(32, 64, 3, 0, dim=3, compress_dim=0, downsample=True)
        self.res3 = ResBlock(32, 64, 3, 0, dim=3, compress_dim=0, downsample=True)
        self.res4 = ResBlock(64, 64, 1, 3, dim=3, compress_dim=0, downsample=True)
        self.res5 = ResBlock(64, 128, 1, 3, dim=3, compress_dim=0, downsample=False)
        self.up1 = UpsampleBlock(256, 128, 4, dim=3, compress_dim=0, upscale=False, conv_dim=2)
        self.up2 = UpsampleBlock(192, 64, 3, dim=3, compress_dim=0, upscale=True, conv_dim=2)
        self.up3 = UpsampleBlock(128, 64, 3, dim=3, compress_dim=0, upscale=True, conv_dim=2)
        # self.up4 = UpsampleBlock(96, 64, 3, dim=3, compress_dim=0, upscale=True, conv_dim=2)
        self.up5 = UpsampleBlock(96, 64, 3, dim=3, compress_dim=0, upscale=True, conv_dim=2)
        self.outc = nn.Conv2d(64, 3, 1, padding=0)
        # self.down_layers = nn.ModuleList([ResBlock(1, 32, 3, 0), ResBlock(32, 32, 3, 0), ResBlock(32, 64, 3, 0), ResBlock(64, 64, 1, 2), ResBlock(64, 128, 1, 2), ResBlock(128, 128, 1, 2), ResBlock(128, 256, 0, 4)])
        # self.up_layers = nn.ModuleList([UpsampleBlock(256, 128, 4), UpsampleBlock(128, 128, 4), UpsampleBlock(128, 128, 4), UpsampleBlock(128, 64, 3), UpsampleBlock(64, 64, 3), UpsampleBlock(64, 3, 3), UpsampleBlock(32, 3, 3)])

    def forward(self, x):
        outputs = []
        x_1, x = self.res1(x)
        # x_2, x = self.res2(x)
        x_3, x = self.res3(x)
        x_4, x = self.res4(x)
        x_5, x = self.res5(x)
        x = x.squeeze(2)
        x = self.up1(x, x_5)
        x = self.up2(x, x_4)
        x = self.up3(x, x_3)
        # x = self.up4(x, x_2)
        x = self.up5(x, x_1)
        x = self.outc(x)
        return x.unsqueeze(2)