import torch
import torch.nn as nn
import torch.nn.functional as F

USE_BIAS=False


class NormalizedComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(NormalizedComplexConv2d, self).__init__()

        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = nn.LayerNorm((kernel_size, kernel_size), elementwise_affine=False)
        self.kernel_size = kernel_size
    def forward(self, input):

        weight_r = self.conv_r.weight - torch.mean(self.conv_r.weight)
        weight_i = self.conv_i.weight - torch.mean(self.conv_i.weight)
        weight_r = weight_r / torch.sum(weight_r ** 2)
        weight_i = weight_i / torch.sum(weight_i ** 2)

        real = torch.nn.functional.conv2d(input.real, weight_r, self.conv_r.bias, self.conv_r.stride, self.conv_r.padding, self.conv_r.dilation, self.conv_r.groups)\
               - torch.nn.functional.conv2d(input.imag, weight_i, self.conv_i.bias, self.conv_i.stride, self.conv_i.padding, self.conv_i.dilation, self.conv_i.groups)
        imag = torch.nn.functional.conv2d(input.imag, weight_r, self.conv_r.bias, self.conv_r.stride, self.conv_r.padding, self.conv_r.dilation, self.conv_r.groups)\
               + torch.nn.functional.conv2d(input.real, weight_i, self.conv_i.bias, self.conv_i.stride, self.conv_i.padding, self.conv_i.dilation, self.conv_i.groups)

        return real + 1j*imag


class NormalizedComplexConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(NormalizedComplexConvTranspose2d, self).__init__()

        self.conv_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.conv_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.norm = nn.LayerNorm((kernel_size, kernel_size), elementwise_affine=False)
    def forward(self, input):

        weight_r = self.conv_r.weight - torch.mean(self.conv_r.weight)
        weight_i = self.conv_i.weight - torch.mean(self.conv_i.weight)
        weight_r = weight_r / torch.sum(weight_r ** 2)
        weight_i = weight_i / torch.sum(weight_i ** 2)

        real = torch.nn.functional.conv_transpose2d(input.real, weight_r, self.conv_r.bias, self.conv_r.stride, self.conv_r.padding, self.conv_r.output_padding, self.conv_r.groups, self.conv_r.dilation)\
               - torch.nn.functional.conv_transpose2d(input.imag, weight_i, self.conv_i.bias, self.conv_i.stride, self.conv_i.padding, self.conv_i.output_padding, self.conv_i.groups, self.conv_i.dilation)
        imag = torch.nn.functional.conv_transpose2d(input.imag, weight_r, self.conv_r.bias, self.conv_r.stride, self.conv_r.padding, self.conv_r.output_padding, self.conv_r.groups, self.conv_r.dilation)\
               + torch.nn.functional.conv_transpose2d(input.real, weight_i, self.conv_i.bias, self.conv_i.stride, self.conv_i.padding, self.conv_i.output_padding, self.conv_i.groups, self.conv_i.dilation)

        return real + 1j*imag


class RadialBasisActivation(nn.Module):
    def __init__(self, in_channels=1, channels=48):
        super(RadialBasisActivation, self).__init__()

        # Scaling and bias for each channel
        self.scale_in = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1)
        self.scale_out = nn.Conv2d(in_channels=channels, out_channels=in_channels, kernel_size=1, bias=False)

    def forward(self, input):
        # Apply a gaussian activation to real and imaginary
        activated_real = torch.exp(-self.scale_in(input.real) ** 2)
        activated_imag = torch.exp(-self.scale_in(input.imag) ** 2)

        # Sum the channels using learned scaling
        out = self.scale_out(activated_real) + 1j * self.scale_out(activated_imag)

        return out


class VarNet( nn.Module):
    def __init__(self, channels=48, kernel_size=11):
        super(VarNet, self).__init__()

        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.channels = channels
        for f in range(channels):
            self.encoding_layers.append(NormalizedComplexConv2d(1,1,kernel_size=kernel_size, padding=kernel_size//2, bias=False))
            self.decoding_layers.append(NormalizedComplexConvTranspose2d(1,1,kernel_size=kernel_size, padding=kernel_size//2, bias=False))
            self.activation_layers.append(RadialBasisActivation(in_channels=1, channels=31))
            #self.activation_layers.append(ComplexReLu())

    def forward(self, image):

        image_temp = torch.zeros_like(image)
        for alayer, elayer, dlayer in zip(self.activation_layers, self.encoding_layers, self.decoding_layers):
            # Encode image
            encoded = elayer(image)

            #Activation
            encoded = alayer(encoded)

            # Decode the layer
            decoded = dlayer(encoded)

            # Add with scale to image
            image_temp += decoded / self.channels


        return image_temp