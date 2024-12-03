import pickle
from typing import Dict


import h5py
import numpy as np
import sigpy as sp
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchsummary
import torch.utils.checkpoint as checkpoint
import torchvision.transforms as transforms

# from utils.ISOResNet import *
from utils.CreateImagePairs import get_smaps, add_gaussian_noise
from utils.unet_componets import *
from utils.model_components import *
from utils.varnet_components_complex import *
from utils.unet import UNet

spdevice = sp.Device(0)


class DnCNN(nn.Module):
    def __init__(self, Nconv=4, Nkernels=16):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(2, Nkernels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(Nconv - 2):
            layers.append(nn.Sequential(nn.Conv2d(Nkernels, Nkernels, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(Nkernels),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(Nkernels, 2, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        residual = self.layiers(inputs)
        return inputs - residual


class DnCNN_dilated(nn.Module):
    def __init__(self, Nconv=7, Nkernels=16, Dilation=None):
        super(DnCNN_dilated, self).__init__()
        if Dilation is None:
            Dilation = [2, 3, 4, 3, 2]
        layers = [nn.Sequential(nn.Conv2d(2, Nkernels, kernel_size=3, stride=1, padding=1, dilation=1),
                                nn.ReLU(inplace=True))]
        for i in range(Nconv - 2):
            layers.append(
                nn.Sequential(nn.Conv2d(Nkernels, Nkernels, kernel_size=3, padding=Dilation[i], dilation=Dilation[i]),
                              nn.BatchNorm2d(Nkernels),
                              nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(Nkernels, 2, kernel_size=3, padding=1, dilation=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(inputs)
        y += residual
        return y



class CNN_shortcut(nn.Module):

    def __init__(self, Nkernels=64):
        super(CNN_shortcut, self).__init__()
        self.conv_in = nn.Sequential(ComplexConv2d(1, Nkernels, kernel_size=3, stride=1, padding=1),
                                     CReLu())
        self.block1 = nn.Sequential(ComplexConv2d(Nkernels, Nkernels, kernel_size=3, stride=1, padding=1),
                                     CReLu())
        self.block2 = nn.Sequential(ComplexConv2d(Nkernels, Nkernels, kernel_size=3, stride=1, padding=1),
                                    CReLu())
        self.block3 = nn.Sequential(ComplexConv2d(Nkernels, Nkernels, kernel_size=3, stride=1, padding=1),
                                    CReLu())

        self.conv_out = ComplexConv2d(Nkernels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x

        # residual
        x = self.conv_in(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv_out(x)

        denoised = identity + x
        return denoised



def sense_adjoint( maps, data):
    im = torch.fft.ifftshift( data, dim=(-2, -1))
    im = torch.fft.ifft2( im, dim=(-2,-1))
    im = torch.fft.fftshift( im, dim=(-2,-1))
    im *= torch.conj(maps)
    im = torch.sum(im, dim=-3, keepdim=True)
    return im

def sense( maps, image ):
    kdata = maps * image
    kdata = torch.fft.ifftshift(kdata, dim=(-2, -1))
    kdata = torch.fft.fft2( kdata, dim=(-2,-1))
    kdata = torch.fft.fftshift(kdata, dim=(-2, -1))
    return kdata


class MoDL(nn.Module):
    '''output: image (sl, 768, 396, 2) '''

    # def __init__(self, inner_iter=10):
    def __init__(self, scale_init=1.0, inner_iter=1, DENOISER='unet', checkpoint_image=False):
        super(MoDL, self).__init__()

        self.checkpoint_image = checkpoint_image
        self.inner_iter = inner_iter
        self.scale_layers = nn.Parameter(scale_init*torch.ones([inner_iter]), requires_grad=True)

        # Options for UNET
        if DENOISER == 'unet':
            self.denoiser = UNet(in_channels=1, out_channels=1, f_maps=64, depth=2,
                                 layer_order=['convolution', 'relu'],
                                 complex_kernel=False, complex_input=True,
                                 residual=True, scaled_residual=True)
            print(self.denoiser)
        elif DENOISER == 'varnet':
            self.varnets = nn.ModuleList()
            for i in range(self.inner_iter):
                self.varnets.append(VarNet())
            self.denoiser = None
        else:
            self.denoiser = CNN_shortcut()

    def checkpoint_fn(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, image, kspace, maps, mask):

        if self.inner_iter == 0:
            pass
        else:

            for i in range(self.inner_iter):
                # Ex
                Ex = sense(maps, image)

                # Ex - d
                diff = (Ex - kspace)*mask
                # diff = Ex - kspace

                # image = image - scale*E.H*(Ex-d)
                image = image - self.scale_layers[i] * sense_adjoint(maps, diff)  # (768, 396)

            PAD=False

            if PAD:
                # Pad to prevent edge effects, circular pad to keep image statistics
                target_size1 = 32 * math.ceil( (64 + image.shape[-1]) / 32)
                target_size2 = 32 * math.ceil( (64 + image.shape[-2]) / 32)
                #print(f'Target size {target_size1} {target_size2}')

                pad_amount1 = (target_size1 - image.shape[-1]) // 2
                pad_amount2 = (target_size2 - image.shape[-2]) // 2
                #print(f'Pad amount {pad_amount1} {pad_amount2}')

                pad_f = (pad_amount1, pad_amount1, pad_amount2, pad_amount2, 0, 0)
                #print(f'Image in {image.shape} pad_f {pad_f}')
                image = nn.functional.pad(image, pad_f)
                #image_complex = nn.functional.pad(image_complex, pad_f)
                # print(f'image_complex reguires grad {image_complex.requires_grad}')
                #print(f'Image in {image.shape} pad_f {pad_f}')

            image = image.unsqueeze(0)  #complex64, (1,1,h,w)
            if self.checkpoint_image:
                image = checkpoint.checkpoint(self.checkpoint_fn(self.denoiser), image)
            else:
                image = self.denoiser(image)

            image = image.squeeze(0)

            if PAD:
                # cropping for UNet
                image = image[:, pad_amount2:-pad_amount2,pad_amount1:-pad_amount1]


        # crop to square
        idxL = int((image.shape[0] - image.shape[1]) / 2)
        idxR = int(idxL + image.shape[1])
        image = image[idxL:idxR,:]

        return image



class EEVarNet_Block(nn.Module):
    def __init__(self, model, scale_init=1.0):
        super(EEVarNet_Block, self).__init__()
        self.model = model
        self.lam = nn.Parameter(scale_init * torch.ones(1), requires_grad=True)

    def forward(self, k0, kspace, mask, encoding_op, decoding_op):

        # Refinement: encoding * CNN(image)
        image = decoding_op.apply(kspace)
        # Padding
        if isinstance(self.model, UNet2D):
            dim = image.ndim
            if dim == 3:
                image = torch.unsqueeze(image, 0)
            image = image.permute(0, -1, 1, 2).contiguous()
            # Pad to prevent edge effects, circular pad to keep image statistics
            target_size1 = 32 * math.ceil((64 + image.shape[-1]) / 32)
            target_size2 = 32 * math.ceil((64 + image.shape[-2]) / 32)

            pad_amount1 = (target_size1 - image.shape[-1]) // 2
            pad_amount2 = (target_size2 - image.shape[-2]) // 2

            pad_f = (pad_amount1, pad_amount1, pad_amount2, pad_amount2)
            # print(pad_f)
            image = nn.functional.pad(image, pad_f, "circular")
            # print(y_pred.shape)
            image = self.model(image)

            # cropping for UNet
            image = image[:, :, pad_amount2:-pad_amount2, pad_amount1:-pad_amount1]
            image = image.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                image = torch.squeeze(image)
        refinement = encoding_op.apply(image)

        # DC
        kspace_complex = torch.view_as_complex(kspace)
        k0_complex = torch.view_as_complex(k0)
        dc_complex = -1 * self.lam * mask * (kspace_complex - k0_complex)
        dc = torch.view_as_real(dc_complex)

        return kspace + dc - refinement


class EEVarNet(nn.Module):
    def __init__(self, num_cascades=12):
        super(EEVarNet, self).__init__()
        self.cascades = nn.ModuleList(
            [EEVarNet_Block(UNet2D(2, 2, depth=4, final_activation='none', f_maps=18, layer_order='cli', EEVarNet=True))
             for _ in range(num_cascades)]
        )


    def forward(self, k0, mask, encoding_op, decoding_op):
        kspace = k0.clone()
        for cascade in self.cascades:
            kspace = cascade(k0, kspace, mask, encoding_op, decoding_op)

        # SOS
        FT_torch = sp.to_pytorch_function(sp.linop.IFFT(tuple(k0.shape[:-1]), axes=range(-3, 0)), input_iscomplex=True, output_iscomplex=True)
        im = FT_torch.apply(kspace)
        imSOS = torch.sum(im**2,dim=0)

        # crop to square
        idxL = int((imSOS.shape[0] - imSOS.shape[1]) / 2)
        idxR = int(idxL + imSOS.shape[1])
        imSOS = imSOS[idxL:idxR,:,:]

        return imSOS


class Denoise_loss(nn.Module):
    def __init__(self, scorenet, in_channels=1, out_channels=1, f_maps=64, depth=3):
        super(Denoise_loss, self).__init__()

        self.scorenet = scorenet
        self.denoiser = UNet(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps, depth=depth,
                             layer_order=['convolution', 'relu'],
                             complex_kernel=False, complex_input=True,
                             residual=True, scaled_residual=True)
        self.preprocess = transforms.Compose([transforms.Resize((224, 224))])


    def forward(self, image, truth, eff=True, sub=True):
        # image and truth needs to be (batch=1, 1, h, w)
        imEst = self.denoiser(image)

        if eff:
            imEst = torch.view_as_real(imEst[0]).permute((0, -1, 1, 2))
            truth = torch.view_as_real(truth[0]).permute((0, -1, 1, 2))
            imEst = self.preprocess(imEst)
            truth = self.preprocess(truth)
            if sub:
                delta_sl = (self.scorenet(imEst - truth) - self.scorenet(torch.zeros_like(imEst))) ** 2
            else:
                delta_sl = (self.scorenet(imEst) - self.scorenet(truth)) ** 2
        else:
            delta_sl = self.scorenet(imEst, truth)
        return delta_sl

