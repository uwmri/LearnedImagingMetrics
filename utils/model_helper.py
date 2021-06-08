from scipy import ndimage

import torch
#from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchinfo import summary
#from torchvision.models.mobilenet import _make_divisible, ConvBNReLU, InvertedResidual

try:
    from efficientnet_pytorch import EfficientNet
except:
    print('Warning efficientnet not found')

from utils.utils import *

import torchvision.transforms.functional as TF
import sigpy as sp
import math
import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=False, val_range=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        img1_abs = torch.abs(img1)
        img2_abs = torch.abs(img2)

        if channel == self.channel and self.window.dtype == img1_abs.dtype:
            window = self.window.to(img1_abs.device).type(img1_abs.dtype)
        else:
            window = create_window(self.window_size, channel).to(img1_abs.device).type(img1_abs.dtype)
            self.window = window
            self.channel = channel

        ssimv = ssim(img1_abs, img2_abs, window=window, window_size=self.window_size,
                    size_average=self.size_average, val_range=self.val_range)
        ssimv = torch.reshape( ssimv, (-1, 1))

        return ssimv



class conv_bn(nn.Module):
    '''conv ->bn + shortcut -> relu'''

    def __init__(self, Nkernels=64, BN=True):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(Nkernels, Nkernels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(Nkernels)
        self.relu = nn.ReLU(inplace=True)
        self.BN = BN

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.BN:
            x = self.norm(x)

        x = self.relu(x)
        return x



import torch.nn.utils.weight_norm as wn

class VarNorm2d(nn.Module):
    def __init__(self):
        super(VarNorm2d,self).__init__()

class VarNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.05, affine=True, track_running_stats=True):
        super(VarNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            var = self.running_var

        input = input / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None]

        return input

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, complex_kernel=False):
        super(ComplexConv2d, self).__init__()
        self.complex_kernel = complex_kernel
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if complex_kernel:
            self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        if self.complex_kernel:
            return apply_complex(self.conv_r, self.conv_i, input)
        else:
            return self.conv_r(input.real) + 1j*self.conv_r(input.imag)

class ComplexAvgPool(nn.Module):
    def __init__(self, pool_rate):
        super(ComplexAvgPool, self).__init__()
        self.pool = nn.AvgPool2d(pool_rate)

    def forward(self, input):
        return self.pool(input.real) + 1j * self.pool(input.imag)


class L2cnnBlock(nn.Module):
    def __init__(self, channels_in=64, channels_out=64, pool_rate=2, bias=False, batch_norm=False, activation=False):
        super(L2cnnBlock, self).__init__()

        if activation:
            self.act_func = nn.ReLU(inplace=True)
        else:
            self.act_func = nn.Identity()

        if batch_norm:
            self.conv1 = nn.Conv2d( channels_in, channels_out, kernel_size=3, padding=1, stride=1, bias=bias)
            self.bn1 =  nn.BatchNorm2d(channels_out)
            self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, stride=1, bias=bias)
            self.bn2 = nn.BatchNorm2d(channels_out)
            self.shortcut = nn.Conv2d( channels_in, channels_out, kernel_size=1, padding=0, stride=1, bias=bias)
        else:
            self.conv1 = ComplexConv2d(channels_in, channels_out, kernel_size=3, padding=1, stride=1, bias=bias)
            self.conv2 = ComplexConv2d(channels_out, channels_out, kernel_size=3, padding=1, stride=1, bias=bias)
            self.shortcut = ComplexConv2d(channels_in, channels_out, kernel_size=1, padding=0, stride=1, bias=bias)
            self.bn1 = VarNorm2d(channels_out)
            self.bn2 = VarNorm2d(channels_out)


        self.pool = ComplexAvgPool(pool_rate)
        self.batch_norm = batch_norm

    def forward(self,x):
        # input x shape torch.Size([48, 1, 396, 396, 2])
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.act_func(x)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = x + shortcut
        x = self.act_func(x)
        x = self.pool(x)
        return x

class saveOutputs():
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

class L2cnn(nn.Module):
    def __init__(self, channel_base=32, channels_in=1,  channel_scale=1, group_depth=5, bias=False, init_scale=1.0):

        super(L2cnn, self).__init__()
        pool_rate = 2
        channels_out = channel_base

        # Connect to output using a linear combination
        self.weight_mse = torch.nn.Parameter(torch.tensor([1.0]).view(1,1))
        self.weight_cnn = torch.nn.Parameter(torch.tensor([1.0]).view(1, 1))
        self.weight_ssim = torch.nn.Parameter(torch.tensor([1.0]).view(1, 1))
        self.scale_weight = nn.Parameter(torch.ones([group_depth]), requires_grad=True)
        self.ssim_op = SSIM( window_size=11, size_average=False, val_range=1)
        #self.scale = nn.Parameter(torch.FloatTensor(init_scale* torch.ones([2])), requires_grad=True)

        self.layers = nn.ModuleList()
        for block in range(group_depth):

            self.layers.append(L2cnnBlock(channels_in, channels_out, pool_rate, bias=bias, activation=False))

            # Update channels
            channels_in = channels_out
            channels_out = channels_out * channel_scale

    def layer_mse(self, x):
        y = x.view(x.shape[0], -1)
        return 1e3*torch.sum(torch.abs(y)**2, dim=1, keepdim=True)**0.5

    def forward(self, input, truth):
        x = input.clone()

        # SSIM (range is -1 to 1)
        #ssim = 2.0 - self.ssim_op(x, truth)

        # print(f'scale shape {self.scale}')
        # for i in range(2):
        #     input[:, i, :, :] *= self.scale[i]

        diff_mag = input - truth
        # if train on 2chan (real and imag) images

        # Update to use sqrt
        #diff_sq = torch.sum( diff ** 2, dim=1, keepdim=True)
        #diff_mag = diff_sq ** (0.5)
        #diff_sq = torch.sum( torch.square(diff), dim=1, keepdim=True)
        #diff_mag = torch.sqrt(diff_sq)

        # Mean square error
        #mse = self.layer_mse(diff_mag)


        # Convolutional pathway with MSE at multiple scales
        for l in self.layers:
            diff_mag = l(diff_mag)
        # print(f'diff_mag shape {diff_mag.shape}') # [64, 32, 1, 1]
        cnn_score = self.layer_mse(diff_mag)    #(64, 1)

        # Combine scores
        #score = torch.abs(self.weight_ssim)*ssim + torch.abs(self.weight_mse)*mse + torch.abs(self.weight_cnn)*cnn_score
        score = cnn_score

        return score

class ResNet2(nn.Module):

    # ResNet for 2 channel.

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, for_denoise=False):
        super(ResNet2, self).__init__()
        self.for_denoise = for_denoise
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # hard code for now. For 396*396 images, output of last block should be 25*25*512
        # self.fc = nn.Linear(512 * 25 * 25, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print((x.shape))
        x = self.avgpool(x)
        #print(x.shape)
        if self.for_denoise:
            return x
        else:
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

# # lost precision while converting between array and pil?
# transform = transforms.Compose([
#      transforms.ToPILImage(),
#      transforms.RandomHorizontalFlip(p=1),
#      transforms.ToTensor(),
#      ])


# class MobileNetV2_2chan(nn.Module):
#     def __init__(self,
#                  num_classes=1,
#                  width_mult=1.0,
#                  inverted_residual_setting=None,
#                  round_nearest=8,
#                  block=None):
#         """
#         MobileNet V2 main class
#
#         Args:
#             num_classes (int): Number of classes
#             width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
#             inverted_residual_setting: Network structure
#             round_nearest (int): Round the number of channels in each layer to be a multiple of this number
#             Set to 1 to turn off rounding
#             block: Module specifying inverted residual building block for mobilenet
#
#         """
#         super(MobileNetV2_2chan, self).__init__()
#
#         if block is None:
#             block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#
#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 # t, c, n, s
#                 [1, 16, 1, 1],
#                 [6, 24, 2, 2],
#                 [6, 32, 3, 2],
#                 [6, 64, 4, 2],
#                 [6, 96, 3, 1],
#                 [6, 160, 3, 2],
#                 [6, 320, 1, 1],
#             ]
#
#         # only check the first element, assuming user knows t,c,n,s are required
#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element list, got {}".format(inverted_residual_setting))
#
#         # building first layer
#         input_channel = _make_divisible(input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
#         features = [ConvBNReLU(2, input_channel, stride=2)]
#         # building inverted residual blocks
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*features)
#
#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )
#
#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
#
#     def _forward_impl(self, x):
#         # This exists since TorchScript doesn't support inheritance, so the superclass method
#         # (this one) needs to have a name other than `forward` that can be accessed in a subclass
#         x = self.features(x)
#         # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
#         x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
#         x = self.classifier(x)
#         return x
#
#     def forward(self, x):
#         return self._forward_impl(x)


def sigpy_image_rotate2( image, theta, verbose=False, device=sp.Device(0)):
    """
    SIgpy based 2D image rotation
    Args:
        image (list or array): List of images to rotate, size [ch,nx,nx]
        theta(float): image rotation
    """
    device = sp.Device(device)
    xp = device.xp

    with device:

        # Random rotation
        r = xp.array([[math.cos(theta), math.sin(theta)],
                     [-math.sin(theta), math.cos(theta)]], np.float32)

        if verbose:
            print(f'Rotation = {r}')

        if isinstance(image, list):
            ny = image[0].shape[1]
            nx = image[0].shape[2]
            nch = image[0].shape[0]
        else:
            ny = image.shape[1]
            nx = image.shape[2]
            nch = image.shape[0]

        # Pad to be account for zero padding
        pad = int( (nx*0.42) // 2)

        ny_pad = ny + 2 * pad
        nx_pad = nx + 2 * pad

        cx = nx // 2
        cy = ny // 2
        cx_pad = nx_pad // 2
        cy_pad = ny_pad // 2

        # Rotate the image coordinates
        y, x = xp.meshgrid(xp.arange(0, ny, dtype=xp.float32),
                           xp.arange(0, nx, dtype=xp.float32),
                           indexing='ij')

        #subtract center of coordinates
        x -= cx
        y -= cx

        # Rotate the coordinates
        coord = xp.stack((y, x), axis=-1)

        if verbose:
            print(f'Coord Shape {coord.shape}')

        coord = xp.expand_dims(coord, -1)
        coord = xp.matmul(r, coord)
        coord = xp.squeeze(coord)
        coord[:, :, 0] += cx_pad
        coord[:, :, 1] += cy_pad


        # Actual rotation
        if isinstance(image, list):
            image_rotated = []
            for imt in image:
                xpad = xp.zeros((ny_pad, nx_pad), dtype=imt.dtype)
                xnew = xp.zeros((nch, ny, nx), dtype=imt.dtype)
                for ch in range(nch):
                    xpad[pad:pad + ny, pad:pad + ny] = imt[ch]
                    xnew[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)
                image_rotated.append(xnew)
        else:
            image_rotated = xp.zeros((nch, ny, nx), dtype=xp.float32)
            xpad = xp.zeros((ny_pad, nx_pad), dtype=xp.float32)
            for ch in range(nch):
                xpad[pad:pad + ny, pad:pad + ny] = image[ch]
                image_rotated[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)

    return image_rotated


def sigpy_image_rotate3( image, theta, verbose=False, device=sp.Device(0), crop=False):
    """
    SIgpy based 2D image rotation

    Args:
        image (list or array): List of images to rotate, size [ch,nx,nx]
        theta(float): image rotation

    """
    device = sp.Device(device)
    xp = device.xp

    with device:

        # Random rotation
        r = xp.array([[math.cos(theta), math.sin(theta)],
                     [-math.sin(theta), math.cos(theta)]], np.float32)

        if verbose:
            print(f'Rotation = {r}')

        if isinstance(image, list):
            ny = image[0].shape[1]
            nx = image[0].shape[2]
            nch = image[0].shape[0]
        else:
            ny = image.shape[1]
            nx = image.shape[2]
            nch = image.shape[0]

        # Pad to be account for zero padding
        pad = int( (nx*0.42) // 2)

        ny_pad = ny + 2 * pad
        nx_pad = nx + 2 * pad

        cx = nx // 2
        cy = ny // 2
        cx_pad = nx_pad // 2
        cy_pad = ny_pad // 2

        # Rotate the image coordinates
        if crop:
            y, x = xp.meshgrid(xp.arange(0, ny_pad, dtype=xp.float32),
                               xp.arange(0, nx_pad, dtype=xp.float32),
                               indexing='ij')
            x -= cx_pad
            y -= cy_pad
        else:
            y, x = xp.meshgrid(xp.arange(0, ny, dtype=xp.float32),
                           xp.arange(0, nx, dtype=xp.float32),
                           indexing='ij')

            #subtract center of coordinates
            x -= cx
            y -= cy

        # Rotate the coordinates
        coord = xp.stack((y, x), axis=-1)

        if verbose:
            print(f'Coord Shape {coord.shape}')

        coord = xp.expand_dims(coord, -1)
        coord = xp.matmul(r, coord)
        coord = xp.squeeze(coord)
        coord[:, :, 0] += cx_pad
        coord[:, :, 1] += cy_pad

        # Actual rotation
        if isinstance(image, list):
            image_rotated = []
            if crop:
                for imt in image:
                    xnew_uncrop = xp.zeros((nch, ny_pad, nx_pad), dtype=xp.float32)
                    xnew = xp.zeros((nch, ny, nx), dtype=xp.float32)
                    for ch in range(nch):
                        xpad = imt[ch]
                        xnew_uncrop[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)
                        xnew[ch] = xnew_uncrop[ch,pad:pad + ny, pad:pad + ny]
                    image_rotated.append(xnew)
            else:
                for imt in image:
                    xpad = xp.zeros((ny_pad, nx_pad), dtype=xp.float32)
                    xnew = xp.zeros((nch, ny, nx), dtype=xp.float32)
                    for ch in range(nch):
                        xpad[pad:pad + ny, pad:pad + ny] = imt[ch]
                        xnew[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)
                    image_rotated.append(xnew)
        else:
            if crop:
                image_rotated_uncrop = xp.zeros((nch, ny_pad, nx_pad), dtype=xp.float32)
                image_rotated = xp.zeros((nch, ny, nx), dtype=xp.float32)
                for ch in range(nch):
                    xpad = image[ch]
                    image_rotated_uncrop[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)
                    image_rotated[ch] = image_rotated_uncrop[ch,pad:pad + ny, pad:pad + ny]
            else:
                image_rotated = xp.zeros((nch, ny, nx), dtype=xp.float32)
                xpad = xp.zeros((ny_pad, nx_pad), dtype=xp.float32)
                for ch in range(nch):
                    xpad[pad:pad + ny, pad:pad + ny] = image[ch]
                    image_rotated[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)




    return image_rotated



class DataGenerator_rank(Dataset):
    def __init__(self, X_1, X_2, X_T, Y, ID, augmentation=False,  roll_magL=-15, roll_magH=15,
                 crop_sizeL=1, crop_sizeH=15, scale_min=0.2, scale_max=2.0, device=sp.Device(0), pad_channels=0):

        '''
        :param X_1: X_1_cnnT/V
        :param X_2: X_2_cnnT/V
        :param X_2: X_2_cnnT/V
        :param Y: labels
        :param transform
        :param augmentation: on/off
        '''
        self.X_1 = X_1
        self.X_2 = X_2
        self.X_T = X_T
        self.Y = Y
        self.ID = ID
        self.augmentation = augmentation
        self.pad_channels = pad_channels

        self.roll_magL = roll_magL
        self.roll_magH = roll_magH

        self.crop_sizeL = crop_sizeL
        self.crop_sizeH = crop_sizeH
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.device = device

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, idx):

        if self.augmentation:
            FLIP = np.ndarray.item(np.random.choice([False, True], size=1, p=[0.5, 0.5]))
            scale = np.random.random()*(self.scale_max-self.scale_min) + self.scale_min
            ROT = True
            ROLL = True
        else:
            FLIP = False
            ROT = False
            ROLL = False
            scale = 1.0

        IDnum = self.ID[idx]
        x1 = scale * self.X_1[IDnum, ...].copy()
        x2 = scale * self.X_2[IDnum, ...].copy()
        xt = scale * self.X_T[IDnum, ...].copy()

        # Push to GPU
        x1 = sp.to_device(x1, self.device)
        x2 = sp.to_device(x2, self.device)
        xt = sp.to_device(xt, self.device)

        # Rotation
        if ROT:
            angle = math.pi*(1- 2.0*np.random.rand())
            x1, x2, xt = sigpy_image_rotate2( [x1, x2, xt], angle, device=self.device)

        xp = self.device.xp

        # flip
        if FLIP:
            flip_axis = np.ndarray.item(np.random.choice([0, 1], size=1, p=[0.5, 0.5]))
            x1 = xp.flip(x1, flip_axis)
            x2 = xp.flip(x2, flip_axis)
            xt = xp.flip(xt, flip_axis)

        if ROLL:
            roll_magLR = np.random.randint(self.roll_magL,self.roll_magH)
            roll_magUD = np.random.randint(self.roll_magL, self.roll_magH)

            x1 = xp.roll(x1, (roll_magLR,roll_magUD),(0,1))
            x2 = xp.roll(x2, (roll_magLR,roll_magUD),(0,1))
            xt = xp.roll(xt, (roll_magLR, roll_magUD), (0, 1))

        # put back to cpu, then send to torch
        x1 = x1.get()
        x2 = x2.get()
        xt = xt.get()
        x1 = sp.to_pytorch(x1, requires_grad=False)
        x2 = sp.to_pytorch(x2, requires_grad=False)
        xt = sp.to_pytorch(xt, requires_grad=False)

        # # make images 3 channel.
        # zeros = torch.zeros(((1,) + x1.shape[1:]), dtype=x1.dtype, device=x1.get_device())
        # for i in range(self.pad_channels):
        #     x1 = torch.cat((x1, zeros), dim=0)
        #     x2 = torch.cat((x2, zeros), dim=0)
        #     xt = torch.cat((xt, zeros), dim=0)

        y = self.Y[idx]

        return x1, x2, xt, y


class MSEmodule(nn.Module):
    def __init__(self):
        super(MSEmodule, self).__init__()

    def forward(self, x, truth):
        y = torch.abs(x - truth)
        y = y.view(y.shape[0], -1)
        truth = truth.view(truth.shape[0], -1)
        return torch.sum(y**2, dim=1, keepdim=True)**0.5


class Classifier(nn.Module):

    def __init__(self, rank):
        super(Classifier,self).__init__()

        self.rank = rank

        self.f = nn.Sequential( nn.Linear(1, 16),
                                 nn.Sigmoid(),
                                 nn.Linear(16, 1))

        self.g = nn.Sequential( nn.Linear(1, 16),
                                 nn.Sigmoid(),
                                 nn.Linear(16, 1))


    def forward(self, image1, image2, imaget):

        #score1 = self.rank(image1, imaget)
        #score2 = self.rank(image2, imaget)

        # Combine the images for batch norm operations
        images_combined = torch.cat([image1, image2], dim=0)    #(batchsize*2, ch=2, 396, 396)
        #print(f'imagecombined shape {images_combined.shape}')
        truth_combined = torch.cat([imaget, imaget], dim=0)

        # Calculate scores
        scores_combined = self.rank(images_combined, truth_combined)
        scores_combined = scores_combined.view(scores_combined.shape[0],-1) #(batchsize*2,1)
        score1 = scores_combined[:image1.shape[0], ...]
        score2 = scores_combined[image1.shape[0]:, ...]

        #score1 = score1.view(score1.shape[0], -1)
        #score2 = score2.view(score2.shape[0], -1)


        # Feed difference to classifier
        d = score1 - score2

        # Train based on symetric assumption
        # P-left = f(x)
        # P-same = g(x), a symetric function g(x) = g(-x)
        # P-right = f(-x)
        px = self.f(d)
        gx = self.g(d) + self.g(-d)
        pnx= self.f(-d)

        d = torch.cat([px, gx, pnx],dim=1)
        d = F.softmax(d, dim=1)      # (BatchSize, 3)

        return d, score1, score2


#backward hook
def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('grad_input norm:', grad_output[0].norm())


def get_sampler(labels):
    """ labels is 1d array (n,) """

    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler


def get_class_weights(labels):
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)], dtype='float32')
    weights = class_sample_count / class_sample_count.max()
    weights = torch.from_numpy(weights)

    return weights.cuda()


class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(1, nc, 1, 1))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, -1.0)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__( self, inplanes, planes,  stride=1, downsample=None, groups=1, base_width=64, dilation = 1, norm_layer= None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = SReLU(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = SReLU(width)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ISOResNet2(nn.Module):

    # ResNet for 2 channel.

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, for_denoise=False):
        super(ISOResNet2, self).__init__()
        self.for_denoise = for_denoise
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn1 = norm_layer(self.inplanes)
        self.relu = SReLU(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # hard code for now. For 396*396 images, output of last block should be 25*25*512
        # self.fc = nn.Linear(512 * 25 * 25, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, input, truth):
        # diff = input - truth
        # # if train on 2chan (real and imag) images
        #
        # x = torch.sum(diff ** 2, dim=1, keepdim=True)
        # x = input.clone()
        x = torch.abs(input - truth)

        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print((x.shape))
        x = self.avgpool(x)
        #print(x.shape)
        if self.for_denoise:
            return x
        else:
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x


    def loss_ortho(self):
        ortho_penalty = []
        cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (7, 7) or m.weight.shape[1] == 3:
                    continue
                o = self.ortho_conv(m)
                cnt += 1
                ortho_penalty.append(o)
        ortho_penalty = sum(ortho_penalty)
        return ortho_penalty

    def ortho_conv(self, m, device='cuda'):
        operator = m.weight
        operand = torch.cat(torch.chunk(m.weight, m.groups, dim=0), dim=1)
        transposed = m.weight.shape[1] < m.weight.shape[0]
        num_channels = m.weight.shape[1] if transposed else m.weight.shape[0]
        if transposed:
            operand = operand.transpose(1, 0)
            operator = operator.transpose(1, 0)
        gram = F.conv2d(operand, operator, padding=(m.kernel_size[0] - 1, m.kernel_size[1] - 1),
                        stride=m.stride, groups=m.groups)
        identity = torch.zeros(gram.shape).to(device)
        identity[:, :, identity.shape[2] // 2, identity.shape[3] // 2] = torch.eye(num_channels).repeat(1, m.groups)
        out = torch.sum((gram - identity) ** 2.0) / 2.0
        return out