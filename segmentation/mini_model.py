import torch
import torch.nn as nn
from resblockunet import ResBlock, UpsampleBlock


# class UNet(nn.Module):
#     def __init__(self, nlayers=2):
#         super(UNet, self).__init__()
#         self.conv1 = nn.Conv3d(1, 64, 3, padding=(1,1,1))
#         self.conv2 = nn.Conv3d(64, 64, 3, padding=(0,1,1))
#         self.pool1 = nn.MaxPool3d((1,2,2))
#         self.conv3 = nn.Conv3d(64, 64, 3, padding=(0,1,1))
#         self.conv4 = nn.Conv3d(64, 64, 3, padding=(0,1,1))
#         self.up_conv1 = nn.ConvTranspose3d(64, 64, 3, stride=2, padding=(0,1,1), output_padding=(0,1,1))
#         self.conv5 = nn.Conv3d(64, 64, 3, padding=(1,1,1))
#         self.conv6 = nn.Conv3d(64, 64, 3, padding=1)
#         self.final_conv = nn.Conv3d(64, 3, 3, padding=(0,1,1))


#     def forward(self, x):
#         x = self.conv1(x)
#         # print("1", x.shape)
#         x = self.conv2(x)
#         # print("2", x.shape)
#         x = self.pool1(x)
#         # print("3", x.shape)
#         x = self.conv3(x)
#         # print("4", x.shape)
#         x = self.conv4(x)
#         # print("5", x.shape)
#         x = self.up_conv1(x)
#         # print("6", x.shape)
#         x = self.conv5(x)
#         # print("7", x.shape)
#         x = self.conv6(x)
#         # print("8", x.shape)
#         x = self.final_conv(x)
#         # print("9", x.shape)
#         return x
        

class UNetResBlocks(nn.Module):
    def __init__(self):
        super(UNetResBlocks, self).__init__()
        self.res1 = ResBlock(1, 32, 3, 0, dim=3, compress_dim=0, downsample=True)
        self.res2 = ResBlock(32, 32, 3, 0, dim=3, compress_dim=0, downsample=True)
        self.res3 = ResBlock(32, 64, 3, 0, dim=3, compress_dim=0, downsample=True)
        self.res4 = ResBlock(64, 64, 1, 3, dim=3, compress_dim=0, downsample=True)
        self.res5 = ResBlock(64, 128, 1, 3, dim=3, compress_dim=0, downsample=False)
        self.up1 = UpsampleBlock(256, 128, 4, dim=3, compress_dim=0, upscale=False, conv_dim=2)
        self.up2 = UpsampleBlock(192, 64, 3, dim=3, compress_dim=0, upscale=True, conv_dim=2)
        self.up3 = UpsampleBlock(128, 64, 3, dim=3, compress_dim=0, upscale=True, conv_dim=2)
        self.up4 = UpsampleBlock(96, 64, 3, dim=3, compress_dim=0, upscale=True, conv_dim=2)
        self.up5 = UpsampleBlock(96, 64, 3, dim=3, compress_dim=0, upscale=True, conv_dim=2)
        self.outc = nn.Conv2d(64, 3, 1, padding=0)
        # self.down_layers = nn.ModuleList([ResBlock(1, 32, 3, 0), ResBlock(32, 32, 3, 0), ResBlock(32, 64, 3, 0), ResBlock(64, 64, 1, 2), ResBlock(64, 128, 1, 2), ResBlock(128, 128, 1, 2), ResBlock(128, 256, 0, 4)])
        # self.up_layers = nn.ModuleList([UpsampleBlock(256, 128, 4), UpsampleBlock(128, 128, 4), UpsampleBlock(128, 128, 4), UpsampleBlock(128, 64, 3), UpsampleBlock(64, 64, 3), UpsampleBlock(64, 3, 3), UpsampleBlock(32, 3, 3)])

    def forward(self, x):
        outputs = []
        x_1, x = self.res1(x)
        x_2, x = self.res2(x)
        x_3, x = self.res3(x)
        x_4, x = self.res4(x)
        x_5, x = self.res5(x)
        x = x.squeeze(2)
        x = self.up1(x, x_5)
        x = self.up2(x, x_4)
        x = self.up3(x, x_3)
        x = self.up4(x, x_2)
        x = self.up5(x, x_1)
        x = self.outc(x)
        return x.unsqueeze(2)