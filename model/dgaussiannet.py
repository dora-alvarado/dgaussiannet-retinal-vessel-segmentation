import torch
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter
from .filters import ApplyDGMF

########################################################################################################################
# This is a pytorch implementation of the D-GaussianNet architecture, from the conference paper:
# - https://link.springer.com/chapter/10.1007%2F978-3-030-72073-5_29
########################################################################################################################

def activation_func(activation):
    # from: https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


def bn_act(in_channels, act=True, p_dropout=0.0):
    layers = [nn.BatchNorm2d(in_channels)]
    if act:
        layers.append(activation_func("relu"))
    layers.append(nn.Dropout2d(p=p_dropout))
    return nn.Sequential(*layers)


def conv_block(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, p_dropout=0.0):
    layers = bn_act(in_channels, p_dropout=p_dropout)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, groups=1)
    return nn.Sequential(*layers.children(), conv)


def shortcut(in_channels, out_channels, padding=0, stride=1):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=padding, stride=stride)
    bn = bn_act(out_channels, act=False)
    return nn.Sequential(conv, bn)


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dgmf_kernel_size=(7,7),padding=1, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.n_sigmas = out_channels#n_sigmas
        self.dgmf_kernel_size = dgmf_kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.conv2 = conv_block(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, p_dropout=0.2)
        self.shortcut1 = shortcut(in_channels, out_channels)
        self.sigmas = nn.Parameter(torch.Tensor(out_channels))
        self.alphas = nn.Parameter(torch.zeros(out_channels))

        dx = torch.tensor(gaussian_filter((torch.rand(*self.dgmf_kernel_size) * 2 - 1), 4., mode="constant", cval=0))  # .to(x.device)
        dy = torch.tensor(gaussian_filter((torch.rand(*self.dgmf_kernel_size) * 2 - 1), 4., mode="constant", cval=0))
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.dy = nn.Parameter(dy, requires_grad=False)

        self.sigmas.data.uniform_(-1, 1) # nn.Parameter(sigmas)#
        self.alphas.data.uniform_(-1, 1)
        self.gmf = ApplyDGMF(self.dgmf_kernel_size, sigmas=self.sigmas, alphas=self.alphas, dx=self.dx, dy=self.dy)

    def forward(self, x):
        self.dx = self.dx.to(x.device)
        self.dy = self.dy.to(x.device)

        residual = x
        self.gmf = ApplyDGMF(self.dgmf_kernel_size, sigmas=self.sigmas,
                             alphas=self.alphas, dx=self.dx, dy=self.dy)
        dgmf1 = self.gmf(1.-x)

        if self.should_apply_shortcut:
            residual = self.shortcut1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual + dgmf1

        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1,
                 p_dropout = 0.0, activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding =padding
        self.stride =stride
        self.activate = activation_func(activation)
        self.conv1 = conv_block(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = conv_block(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1, p_dropout=0.2)
        self.dropout = nn.Dropout2d(p=p_dropout)
        self.shortcut1 = shortcut(in_channels, out_channels, stride=stride)

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut1(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class Upsample_Concat(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def forward(self, x, xskip):
        x = self.upsample(x)
        x = torch.cat([x, xskip], dim=1)
        return x


def double_conv(in_channels, out_channels, p_dropout=0.2, padding=1,
                batchnorm=True, activation=True):
    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=padding, groups=1)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Dropout2d(p=p_dropout))
    layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=padding, groups=1))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class DGaussianNet(nn.Module):
    def __init__(self, n_channels, n_classes=2, seg_grad_cam=False):
        super().__init__()
        self.n_classes = n_classes
        self.seg_grad_cam = seg_grad_cam
        # Encoder
        self.stem1 = Stem(n_channels, 16)
        self.down1 = ResidualBlock(16, 32, stride=2)
        self.down2 = ResidualBlock(32, 64, stride=2)
        self.down3 = ResidualBlock(64, 128, stride=2)
        # Bridge
        self.middle1 = conv_block(128, 256, stride=1)
        self.middle2 = conv_block(256, 256, stride=1, p_dropout=0.2)
        # Decoder
        self.up1 = ResidualBlock(256+64, 128)
        self.up2 = ResidualBlock(128+32, 64)
        self.up3 = ResidualBlock(64+16, 32)
        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stem2 = Stem(32, 16)
        self.final = nn.Sequential(nn.Conv2d(16, self.n_classes, 1),)
        self.conv0 = None
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None

    def forward(self,  x):
        # Encoder
        self.conv0 = self.stem1(x)
        self.conv1 = self.down1(self.conv0)
        self.conv2 = self.down2(self.conv1)
        self.conv3 = self.down3(self.conv2)
        # Bridge
        x = self.middle1(self.conv3)
        x = self.middle2(x)
        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, self.conv2], dim=1)
        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, self.conv1], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, self.conv0], dim=1)
        x = self.up3(x)
        x = self.stem2(x)
        if not self.seg_grad_cam:
            x = self.final(x)
        return x


