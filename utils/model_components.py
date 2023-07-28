import torch
import torch.nn as nn
import torch.nn.functional as F
import math

USE_BIAS=False


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

# class ComplexReLu(nn.Module):
#     def __init__(self,  inplace=False):
#         super(ComplexReLu, self).__init__()
#         self.inplace = inplace
#
#     def forward(self, input):
#         mag = torch.abs( input)
#         return torch.nn.functional.relu(mag, inplace=self.inplace).type(torch.complex64)/(mag+1e-6)*input

class CReLu(nn.Module):
    def __init__(self):
        super(CReLu, self).__init__()

    def forward(self, input):
        return torch.nn.functional.relu(input.real, inplace=False) + 1j*torch.nn.functional.relu(input.imag, inplace=False)

class SReLU(nn.Module):
    """Shifted ReLU from https://arxiv.org/pdf/2006.16992.pdf"""

    def __init__(self, nc):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(1, nc, 1, 1))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, 0.0)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias


class CReLU_bias(nn.Module):
    def __init__(self, nc):
        super(CReLU_bias, self).__init__()
        #self.act_func = SReLU(nc)
        self.srelu_bias = nn.Parameter(torch.Tensor(1, nc, 1, 1))
        self.srelu_relu = nn.ReLU(inplace=False)
        nn.init.constant_(self.srelu_bias, 0.0)

    def forward(self, x):
        #return self.act_func(x.real) + 1j*self.act_func(x.imag)
        activated_re = self.srelu_relu(x.real - self.srelu_bias) + self.srelu_bias
        activated_im = self.srelu_relu(x.imag - self.srelu_bias) + self.srelu_bias
        return activated_re + 1j * activated_im


class ZReLU(nn.Module):
    def __init__(self):
        super(ZReLU, self).__init__()

    def forward(self, input):
        phase = torch.angle(input)
        zeros = torch.zeros_like(input)

        le = torch.le(phase, math.pi/2)
        input = torch.where(le, input, zeros)
        ge = torch.ge(phase, 0)
        input = torch.where(ge, input, zeros)
        return input


class modReLU(nn.Module):
    '''
    A PyTorch module to apply relu activation on the magnitude of the signal. Phase is preserved
    '''
    def __init__(self,in_channels=None, ndims=2):
        super(modReLU, self).__init__()
        self.act = nn.ReLU(inplace=False)
        shape = (1, in_channels) + tuple(1 for _ in range(ndims))
        self.bias = nn.Parameter(torch.zeros(shape), requires_grad=True)

    def forward(self, input):
        mag = input.abs()
        return self.act(mag+self.bias) * input / (mag + torch.finfo(mag.dtype).eps)


class SCReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc):
        super(SCReLU, self).__init__()
        self.srelu_bias_re = nn.Parameter(torch.Tensor(1, nc, 1, 1))
        self.srelu_bias_im = nn.Parameter(torch.Tensor(1, nc, 1, 1))

        self.srelu_relu = nn.ReLU(inplace=False)
        nn.init.constant_(self.srelu_bias_re, -1.0)
        nn.init.constant_(self.srelu_bias_im, -1.0)


    def forward(self, x):
        activated_re = self.srelu_relu(x.real - self.srelu_bias_re) + self.srelu_bias_re
        activated_im = self.srelu_relu(x.imag - self.srelu_bias_im) + self.srelu_bias_im
        return activated_re + 1j*activated_im


class ComplexLeakyReLu(nn.Module):
    def __init__(self,  inplace=False):
        super(ComplexLeakyReLu, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return torch.nn.functional.leaky_relu(input.real, negative_slope=0.01, inplace=self.inplace) + \
               1j*torch.nn.functional.leaky_relu(input.imag, negative_slope=0.01, inplace=self.inplace)

class ComplexELU(nn.Module):
    def __init__(self,  inplace=False):
        super(ComplexELU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        mag = torch.abs(input)
        return torch.nn.functional.elu(mag, inplace=self.inplace).type(torch.complex64)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, groups=1, ndims=2):
    if ndims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)
    elif ndims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)
    else:
        raise ValueError(f'Convolution  must be 2D or 3D passed {ndims}')


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=USE_BIAS, complex_kernel=False):
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


class ComplexConv(nn.Module):
    '''
    This convolution supporting complex inputs and complex kernels and 2D or 3D convolutions.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all six sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        complex_kernel (bool, optional): If ``True`` declares the kernel as complex and applies it so. Otherwise
            the kernel will be real valued. Default: ``True``
        ndims (int, optional): 2 or 3 specifying 2D or 3D convolutions. Default: 2
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, groups=1, bias=False, complex_kernel=False, ndims=2):
        super(ComplexConv, self).__init__()
        self.complex_kernel = complex_kernel

        if ndims == 2:
            self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                    padding_mode=padding_mode)
            if complex_kernel:
                self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                        padding_mode=padding_mode)
        elif ndims == 3:
            self.conv_r = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                    padding_mode=padding_mode)
            if complex_kernel:
                self.conv_i = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                        padding_mode=padding_mode)
        else:
            raise ValueError(f'Convolutions must be 2D or 3D passed {ndims}')

    def forward(self, input):
        if self.complex_kernel:
            return apply_complex(self.conv_r, self.conv_i, input)
        else:
            return self.conv_r(input.real) + 1j * self.conv_r(input.imag)


class ComplexConvTranspose(nn.Module):
    '''

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        complex_kernel (bool, optional): If ``True`` declares the kernel as complex and applies it so. Otherwise
            the kernel will be real valued. Default: ``True``
        ndims (int, optional): 2 or 3 specifying 2D or 3D convolutions. Default: 2

    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros',
                 complex_kernel=False, ndims=2):
        super(ComplexConvTranspose, self).__init__()

        self.complex_kernel = complex_kernel

        if ndims == 2:
            self.conv_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                             output_padding, groups, bias, dilation, padding_mode)
            if self.complex_kernel:
                self.conv_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)
        elif ndims == 3:
            self.conv_r = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding,
                                             output_padding, groups, bias, dilation, padding_mode)
            if self.complex_kernel:
                self.conv_i = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)
        else:
            raise ValueError(f'Convolution transpose must be 2D or 3D passed {ndims}')

    def forward(self, input):
        if self.complex_kernel:
            return apply_complex(self.conv_r, self.conv_i, input)
        else:
            return self.conv_r(input.real) + 1j * self.conv_r(input.imag)


class ComplexDepthwiseSeparableConv(nn.Module):
    '''

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        complex_kernel (bool, optional): If ``True`` declares the kernel as complex and applies it so. Otherwise
            the kernel will be real valued. Default: ``True``
        ndims (int, optional): 2 or 3 specifying 2D or 3D convolutions. Default: 2

    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, groups=1, bias=False, complex_kernel=False, ndims=2):
        super(ComplexDepthwiseSeparableConv, self).__init__()

        self.depthwise = ComplexConv(in_channels, in_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     padding_mode=padding_mode,
                                     dilation=dilation,
                                     groups=in_channels,
                                     bias=bias,
                                     complex_kernel=complex_kernel,
                                     ndims=ndims)

        self.pointwise = ComplexConv(in_channels, out_channels,
                                     kernel_size=1,
                                     stride=stride,
                                     padding=0,
                                     padding_mode=padding_mode,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias,
                                     complex_kernel=complex_kernel,
                                     ndims=ndims)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



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
            n = input.numel() / input.size()
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


class VarNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.05, affine=True, track_running_stats=True):
        super(VarNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

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
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size()
            with torch.no_grad():
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            var = self.running_var

        input = input / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None]

        return input


class ComplexDropout2D(torch.nn.Dropout2d):
    def forward(self, input):

        in_shape = input.shape

        # Put the real and imaginary at the last dimension
        tensor = torch.stack([input.real, input.imag], dim=-1)

        # Apply dropout on the tensor Nc x Ny x (Nx*2)
        output = super().forward(tensor.reshape(in_shape[:-1] + (in_shape[-1]*2,) ))

        # Reshape back to starting shape x 2
        output = output.reshape(in_shape + (2,))

        return output[...,0] + 1j*output[...,1]


class ComplexAvgPool(nn.Module):
    def __init__(self, pool_rate):
        super(ComplexAvgPool, self).__init__()
        self.pool = nn.AvgPool2d(pool_rate)

    def forward(self, input):
        return self.pool(input.real) + 1j * self.pool(input.imag)




