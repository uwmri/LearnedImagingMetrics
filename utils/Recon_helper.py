import pickle
from typing import Dict

import matplotlib

import h5py
import numpy as np
import sigpy as sp
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchsummary
import torch.utils.checkpoint as checkpoint
from utils.model_helper import *
from utils.CreateImagePairs import get_smaps, add_gaussian_noise
from utils.unet_componets import *
from utils.unet_components_complex import *
from utils.varnet_components_complex import *
from mri_unet.unet import UNet

spdevice = sp.Device(0)

class SubtractArray(sp.linop.Linop):
    """Subtract array operator, subtracts a given array allowing composed operator
    Args:
        shape (tuple of ints): Input shape
        x: array to subtract from the input
    """

    def __init__(self, x):
        self.x = x
        super().__init__(x.shape, x.shape)

    def _apply(self, input):
        return (input - self.x)

    def _adjoint_linop(self):
        return self

import random
class DataGeneratorReconSlices(Dataset):

    def __init__(self,path_root, rank_trained_on_mag=False, data_type=None, case_name=False):

        '''
        input: mask (768, 396) complex64
        output: all complex numpy array
                fully sampled kspace (rectangular), truth image (square), smaps (rectangular)
        '''


        self.scans = [f for f in os.listdir(path_root) if os.path.isfile(os.path.join(path_root, f))]
        random.shuffle(self.scans)

        print(f'Found {len(self.scans)} from {path_root}')

        self.len = len(self.scans)
        self.data_folder = path_root
        self.data_type = data_type
        self.rank_trained_on_mag = rank_trained_on_mag
        self.case_name = case_name


    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        fname = self.scans[idx]

        with h5py.File(os.path.join(self.data_folder,fname),'r') as hf:
            if self.data_type == 'smap16':
                smaps = np.array(hf['smaps'])
                smaps = smaps.view(np.int16).astype(np.float32).view(np.complex64)
                smaps /= 32760
                #print(smaps.shape)
            else:
                smaps = np.array(hf['smaps'])

            #t = time.time()
            kspace = np.array(hf['kspace'])
            kspace = zero_pad3D(kspace)  # array (sl, coil, 768, 396)
            kspace /= np.max(np.abs(kspace))/np.prod(kspace.shape[-2:])

        # Copy to torch
        kspace = torch.from_numpy(kspace)
        smaps = torch.from_numpy(smaps)

        if not self.case_name:
            return smaps, kspace
        else:
            return smaps, kspace, fname

class DataGeneratorRecon(Dataset):

    def __init__(self,path_root, h5file, indices=None, rank_trained_on_mag=False, data_type=None, case_name=False,
                 DiffCaseEveryEpoch=False):

        '''
        input: mask (768, 396) complex64
        output: all complex numpy array
                fully sampled kspace (rectangular), truth image (square), smaps (rectangular)
        '''

        # scan path+file name
        #with open(os.path.join(path_root, scan_list), 'rb') as tf:
        #    self.scans = pickle.load(tf)

        self.hf = h5py.File(name=os.path.join(path_root, h5file), mode='r')
        self.scans = [f for f in self.hf.keys()]
        self.num_allcases = len((self.scans))
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.num_allcases)

        self.len = len(self.indices)

        self.data_type = data_type

        self.rank_trained_on_mag = rank_trained_on_mag

        self.case_name = case_name
        self.DiffCaseEveryEpoch = DiffCaseEveryEpoch

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        actual_idx = self.indices[idx]

        #print(f'Load {self.scans[actual_idx]}')
        #t = time.time()
        if self.data_type == 'smap16':
            smaps = np.array(self.hf[self.scans[actual_idx]]['smaps'])
            smaps = smaps.view(np.int16).astype(np.float32).view(np.complex64)
            smaps /= 32760
            #print(smaps.shape)
        else:
            smaps = np.array(self.hf[self.scans[actual_idx]]['smaps'])

        #smaps = complex_2chan(smaps)
        #print(f'Get smap ={time.time()-t}, {smaps.dtype} {smaps.shape}')

        #truth = None
        # t = time.time()
        # truth = np.array(self.hf[self.scans[idx]]['truths'])
        # #truth = complex_2chan(truth)
        # #if self.rank_trained_on_mag:
        # #    truth = np.sqrt(np.sum(np.square(truth), axis=-1, keepdims=True))
        # truth = zero_pad_truth(truth)
        # truth = torch.from_numpy(truth)     # ([16, 768, 396, ch])
        #
        # # normalize truth to 0 and 1
        # # maxt_truth shape  (sl, 1, 1, 1)
        # max_truth, _ = torch.max(truth, dim=1, keepdim=True)
        # max_truth, _ = torch.max(max_truth, dim=2, keepdim=True)
        # max_truth, _ = torch.max(max_truth, dim=3, keepdim=True)
        # truth /= max_truth
        # #print(f'Get truth {time.time() - t} {truth.dtype} {truth.shape}')

        #t = time.time()
        kspace = np.array(self.hf[self.scans[actual_idx]]['kspace'])
        kspace = zero_pad4D(kspace)  # array (sl, coil, 768, 396)
        kspace /= np.max(np.abs(kspace))/np.prod(kspace.shape[-2:])

        # Copy to torch
        kspace = torch.from_numpy(kspace)
        smaps = torch.from_numpy(smaps)

        #max_truth = torch.unsqueeze(max_truth, -1)
        #kspace = complex_2chan(kspace)  # (slice, coil, h, w, 2)
        #kspace /= max_truth

        if not self.case_name:
            return smaps, kspace
        else:
            return smaps, kspace, self.scans[actual_idx]

class DataGeneratorDenoise(Dataset):
    def __init__(self, path_root, scan_list, h5file):
        # scan path+file name
        with open(os.path.join(path_root, scan_list), 'rb') as tf:
            self.scans = pickle.load(tf)
        self.hf = h5py.File(name=os.path.join(path_root, h5file), mode='r')

        # iterate over scans
        self.len = len(self.hf)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        truth = self.hf[self.scans[idx]]['truths']
        truth = truth[:]  # (Nslice, 396,396) complex64
        noisy = np.zeros(truth.shape, dtype=truth.dtype)
        for i in range(truth.shape[0]):
            noisy[i, ...] = add_gaussian_noise(truth[i, ...], prob=1, level=1e4, mode=0, mean=0)
            # plt.imshow(np.abs(noisy[i,...]))
            # plt.show()

        truth = complex_2chan(truth)
        truth = torch.from_numpy(truth)
        truth = truth.permute(0, -1, 1, 2)

        noisy = complex_2chan(noisy)
        noisy = torch.from_numpy(noisy)
        noisy = noisy.permute(0, -1, 1, 2)

        return truth, noisy


class DataIterator():
    def __init__(self, loader):
        self.loader = iter(loader)

    def next(self):
        try:
            smaps, im, kspaceU = next(self.loader)
            # im = im.cuda()
        except StopIteration:
            smaps = None
            im = None
            kspaceU = None

        return smaps, im, kspaceU


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_smaps, self.next_im, self.next_kspaceU = next(self.loader)
        except StopIteration:
            self.next_smaps = None
            self.next_im = None
            self.next_kspaceU = None
            return
        with torch.cuda.stream(self.stream):
            # self.next_kspaceU = torch.squeeze(self.next_kspaceU)
            self.next_im = torch.squeeze(self.next_im)

            # self.next_kspaceU = self.next_kspaceU.cuda(non_blocking=True)
            self.next_im = self.next_im.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        smaps = self.next_smaps
        im = self.next_im
        kspaceU = self.next_kspaceU

        if smaps is not None:
            smaps.record_stream(torch.cuda.current_stream())
        if im is not None:
            im.record_stream(torch.cuda.current_stream())
        if kspaceU is not None:
            kspaceU.record_stream(torch.cuda.current_stream())

        self.preload()
        return smaps, im, kspaceU


# plot a training/val set
def sneakpeek(dataset, Ncoils=20, rank_trained_on_mag=True):
    idx = np.random.randint(len(dataset))
    checksmaps, checktruth, checkkspace = dataset[idx]
    Nslice = checktruth.shape[0]
    slice_num = np.random.randint(Nslice)
    coil_num = np.random.randint(20)

    checksmaps = chan2_complex(checksmaps)
    checksmaps = checksmaps[slice_num]
    checkkspace = chan2_complex(checkkspace.numpy())

    if not rank_trained_on_mag:
        checktruth = chan2_complex(checktruth.numpy())
        checktruth = np.abs(checktruth)
    plt.imshow(checktruth[slice_num], cmap='gray')
    plt.show()
    plt.imshow(np.log(np.abs(checkkspace[slice_num, coil_num, ...]) + 1e-5), cmap='gray')
    plt.show()

    plt.figure(figsize=(10, 10))
    for m in range(int(np.ceil(Ncoils / 4))):
        for n in range(4):
            plt.subplot(int(np.ceil(Ncoils / 4)), 4, (m * 4 + n + 1))
            plt.imshow(np.abs(checksmaps[(m * 4 + n), :, :]), cmap='gray')
            plt.axis('off')
    plt.show()


# MSE loss
def mseloss_fcn(output, target):
    mse = torch.abs(output - target)
    mse = mse.view(mse.shape[0], -1)
    mse = torch.sum(mse ** 2, dim=1, keepdim=True) ** 0.5

    return mse

def mseloss_fcn0(output, target):
    loss = torch.mean(torch.abs(output - target) ** 2) ** 0.5
    return loss

# TODO: how do i save encoder during training
def loss_fcn_onenet(noisy, output, target, projector, encoder, discriminator, discriminator_l, lam1=1, lam2=1, lam3=1,
                    lam4=-1, lam5=-1):
    loss1 = lam1 * torch.mean((target - projector(target)) ** 2)
    loss2 = lam2 * torch.mean((target - output) ** 2)
    loss3 = lam3 * torch.mean((noisy - output) ** 2)

    cross_entropy = nn.CrossEntropyLoss()
    loss4 = lam4 * cross_entropy(discriminator_l(encoder(target)), discriminator(encoder(noisy)))
    loss5 = lam5 * cross_entropy(discriminator(target), discriminator(output))

    return loss1 + loss2 + loss3 + loss4 + loss5


# learned metrics loss
def learnedloss_fcn(output, target, scoreModel, rank_trained_on_mag=False, augmentation=False):

    if output.ndim == 2:
        output = output.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    if output.ndim == 3:
        output = torch.unsqueeze(output, 0)
        target = torch.unsqueeze(target, 0)


    #if rank_trained_on_mag and output.shape[-1] == 2:
    #    output = torch.sqrt(torch.sum(output**2, axis=-1, keepdims=True))

    #output = torch.abs(output.unsqueeze(0).unsqueeze(0))
    #target = torch.abs(target.unsqueeze(0).unsqueeze(0))

    # output = crop_im(output)
    # target = crop_im(target)
    #output = output.permute(0, -1, 1,2)
    #target = target.permute(0, -1, 1,2)

    Nslice = output.shape[0]


    # if not rank_trained_on_mag:
    #     # add a zero channel since ranknet expect 3chan
    #     zeros = torch.zeros(((Nslice, 1,) + output.shape[2:]), dtype=output.dtype)
    #     zeros = zeros.cuda()
    #     output = torch.cat((output, zeros), dim=1)

    #
    #     target = torch.cat((target, zeros), dim=1)      # (batch=1, 3, 396, 396)

    delta = 0
    count = 1.0
    for sl in range(Nslice):

        output_sl = torch.unsqueeze(output[sl], 0)
        target_sl = torch.unsqueeze(target[sl], 0)

        if augmentation:

            shiftx = np.random.choice([-1, 0, 1])
            shifty = np.random.choice([-1, 0, 1])

            output_sl = torch.roll(output_sl, (shiftx, shifty), dims=(-2,-1))
            target_sl = torch.roll(target_sl, (shiftx, shifty), dims=(-2,-1))

            delta_sl = scoreModel(output_sl, target_sl)
            delta += delta_sl
            count += 1.0

            # Flip Up/Dn
            output_sl2 = torch.flip(output_sl, dims=(-1,))
            target_sl2 = torch.flip(target_sl, dims=(-1,))
            # delta_sl = torch.abs(scoreModel(output_sl2, target_sl2) - bias)
            delta_sl = scoreModel(output_sl2, target_sl2)
            delta += delta_sl
            count += 1.0

            # Flip L/R
            output_sl2 = torch.flip(output_sl, dims=(-2,))
            target_sl2 = torch.flip(target_sl, dims=(-2,))
            # delta_sl = torch.abs(scoreModel(output_sl2, target_sl2) - bias)
            delta_sl = scoreModel(output_sl2, target_sl2)
            delta += delta_sl
            count += 1.0
        else:
            delta_sl = scoreModel(output_sl, target_sl)
            delta += delta_sl
            count += 1.0

    delta /= count

    return delta



# perceptual loss
class PerceptualLoss_VGG16(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss_VGG16, self).__init__()

        # ReLU_2
        self.net = torchvision.models.vgg16(pretrained=True).features[:9].eval()
        # blocks = []
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        # # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        # for bl in blocks:
        #     for p in bl:
        #         p.requires_grad = False
        # self.blocks = torch.nn.ModuleList(blocks)
        for param in self.net.parameters():
            param.requires_grad = False

        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)).cuda()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):

        # truth (1, 768, 396, 1). target (1,768, 396, 2)
        if input.ndim == 3:
            input = torch.unsqueeze(input, 0)
            target = torch.unsqueeze(target, 0)

        inputabs = torch.sqrt(input[:, :, :, 0] ** 2 + input[:, :, :, 1] ** 2)
        inputabs = torch.unsqueeze(inputabs, dim=3)
        input = torch.cat((inputabs, inputabs, inputabs), dim=3)
        shape3 = input.shape

        # targetabs = torch.sqrt(target[:, :, :, 0] ** 2 + target[:, :, :, 1] ** 2)
        # targetabs = torch.unsqueeze(targetabs, dim=3)
        target = torch.cat((target, target, target), dim=3)

        # normalize to (0,1)
        input = input.view(input.shape[0], -1)
        input -= input.min(1, keepdim=True)[0]
        input /= input.max(1, keepdim=True)[0]
        input = input.view(shape3)

        target = target.view(target.shape[0], -1)
        target -= target.min(1, keepdim=True)[0]
        target /= target.max(1, keepdim=True)[0]
        target = target.view(shape3)

        input = input.permute(0, -1, 1, 2)  # to (slice, 3, 396, 396)
        target = target.permute(0, -1, 1, 2)

        self.mean.cuda()
        self.std.cuda()
        self.net.cuda()

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        # # mean of perceptual loss from j-th ReLU
        # for block in self.blocks:
        #     input = block(input)
        #     target = block(target)
        #     norm = input.shape[1] * input.shape[2] * input.shape[3]
        #     loss = ((input - target)**2)/norm
        #
        #     loss += loss
        # loss /= len(self.blocks)

        loss = (self.net(input) - self.net(target)) ** 2
        norm = loss.shape[1] * loss.shape[2] * loss.shape[3]
        loss /= norm

        loss = torch.sum(loss.contiguous().view(loss.shape[0], -1), -1)  # shape (NSlice)

        return torch.mean(loss)


# patch GAN loss
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        # no need to use bias as BatchNorm2d has affine parameters

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def loss_GAN(input, target, discriminator):

    if input.ndim == 3:
        input = torch.unsqueeze(input, 0)
        target = torch.unsqueeze(target, 0)

    Nslice = target.shape[0]
    input = input.permute(0, -1, 1, 2)      # (sl, ch, h, w)
    target = target.permute(0, -1, 1, 2)
    Dtruth = torch.sum(torch.mean(discriminator(target.contiguous()).view(Nslice, -1), dim=1)) / 8
    Drecon = torch.sum(torch.mean(discriminator(input.contiguous()).view(Nslice, -1), dim=1)) / 8

    return -torch.log(Dtruth.abs_()) + torch.log(1.0 - Drecon.abs_())


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
        # layers.append(nn.BatchNorm2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        residual = self.layers(inputs)
        return inputs - residual


class DnCNN_dilated(nn.Module):
    def __init__(self, Nconv=7, Nkernels=16, Dilation=None):
        super(DnCNN_dilated, self).__init__()
        if Dilation is None:
            Dilation = [2, 3, 4, 3, 2]
            # Dilation = [2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        layers = [nn.Sequential(nn.Conv2d(2, Nkernels, kernel_size=3, stride=1, padding=1, dilation=1),
                                nn.ReLU(inplace=True))]
        for i in range(Nconv - 2):
            layers.append(
                nn.Sequential(nn.Conv2d(Nkernels, Nkernels, kernel_size=3, padding=Dilation[i], dilation=Dilation[i]),
                              nn.BatchNorm2d(Nkernels),
                              nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(Nkernels, 2, kernel_size=3, padding=1, dilation=1))
        # layers.append(nn.BatchNorm2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(inputs)
        y += residual
        return y


class conv_bn(nn.Module):
    '''conv ->bn + shortcut -> relu'''

    def __init__(self, Nkernels=64, BN=True):
        super(conv_bn, self).__init__()
        self.conv = ComplexConv2d(Nkernels, Nkernels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(Nkernels)
        self.relu = CReLu()
        self.BN = BN

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.BN:
            x = self.norm(x)

        x += identity
        x = self.relu(x)
        return x


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
        # x = self.block4(x)
        x = self.conv_out(x)

        denoised = identity + x
        return denoised


class Projector(nn.Module):
    # TODO: add channel-wise fully connected.
    # Based on https://arxiv.org/pdf/1703.09912.pdf. Replaced virtual BN with regular BN.

    def __init__(self, ENC=False):
        super(Projector, self).__init__()
        self.layer1 = self._make_layer_enc(2, 64, kernel_size=4, stride=1)
        self.layer2 = self._make_layer_enc(64, 128, kernel_size=4, stride=1)
        self.layer3 = self._make_layer_enc(128, 256, kernel_size=4, stride=2)
        self.layer4 = self._make_layer_enc(256, 512, kernel_size=4, stride=2)
        self.layer5 = self._make_layer_enc(512, 1024, kernel_size=4, stride=2)
        self.context_cfc = nn.Conv1d(1024, 1024, kernel_size=2, groups=1024)
        self.context_conv = self._make_layer_enc(1024, 1024, kernel_size=2, stride=1)

        self.layer6 = self._make_layer_dec(1024, 512, kernel_size=4, stride=2)
        self.layer7 = self._make_layer_dec(512, 256, kernel_size=4, stride=2)
        self.layer8 = self._make_layer_dec(256, 128, kernel_size=4, stride=2)
        self.layer9 = self._make_layer_dec(128, 64, kernel_size=4, stride=1)
        self.layer10 = self._make_layer_dec(64, 2, kernel_size=4, stride=1)

        self.ENC = ENC

    def _make_layer_enc(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0)):
        layers = []
        layers.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ELU(inplace=True)))
        return nn.Sequential(*layers)

    def _make_layer_dec(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0)):
        layers = []
        layers.append(nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ELU(inplace=True)))
        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.context_cfc(x)
        # x = self.context_conv(x)
        if self.ENC:
            return x
        else:
            # decoder
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x = self.layer9(x)
            x = self.layer10(x)

        return x

class Bottleneck(nn.Module):

    def __init__(self, type, inplanes, stride):
        super(Bottleneck, self).__init__()
        self.shortcut = self._make_shortcut(type, inplanes=inplanes)
        self.conv = self._make_conv_layer(type, inplanes=inplanes, stride=stride, channel_compress_ratio=4)

    def _make_conv_layer(self, type, inplanes, stride, channel_compress_ratio=4):
        if type == 'same' or type == 'quarter':
            output_channel = inplanes
        else:
            output_channel = int(inplanes * 2)
        bottleneck_channel = int(output_channel / channel_compress_ratio)

        layers = []
        layers.append(nn.Sequential(nn.BatchNorm2d(inplanes),
                                    nn.ELU(inplace=True),
                                    nn.Conv2d(inplanes, bottleneck_channel, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(bottleneck_channel),
                                    nn.ELU(inplace=True),
                                    nn.Conv2d(bottleneck_channel, bottleneck_channel, kernel_size=3, stride=1,
                                              padding=1),
                                    nn.BatchNorm2d(bottleneck_channel),
                                    nn.ELU(inplace=True),
                                    nn.Conv2d(bottleneck_channel, output_channel, kernel_size=1, stride=1)))
        return nn.Sequential(*layers)

    def _make_shortcut(self, type, inplanes):
        if type == 'same':
            return nn.Identity()
        elif type == 'quarter':
            return nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=2)
        else:
            output_channel = inplanes * 2
            return nn.Conv2d(inplanes, output_channel, kernel_size=1, stride=2)

    def forward(self, x):
        short = self.shortcut(x)
        print(short.shape)
        conv = self.conv(x)
        print(conv.shape)
        return short + conv


class ClassifierD(nn.Module):
    # Based on https://arxiv.org/pdf/1703.09912.pdf
    '''
    Input: layers: length-4 list, default is [3,4,6,3] according to paper
    TODO: HOW to make fc dynamically change the number of input features based on number of slices.
            Right now need to pass it.
    '''

    def __init__(self, Nlayers=None, Nslices=2):
        super(ClassifierD, self).__init__()
        if Nlayers is None:
            Nlayers = [3, 4, 6, 3]
        self.net = self._build_block(Nlayers=Nlayers)
        self.Nslices = Nslices
        self.fc = nn.Linear(1024 * 25 * 25 * self.Nslices, 1)

    def _build_block(self, Nlayers, inplanes=64):
        layers = []
        layers.append(nn.Conv2d(2, inplanes, kernel_size=4, stride=1, padding=2))
        for i in range(len(Nlayers)):
            # print(f'half, inplanes={int(inplanes*(2**i))}')
            layers.append(Bottleneck(type='half', inplanes=int(inplanes * (2 ** i)), stride=2))
            for j in range(Nlayers[i]):
                # print(f'same, inplanes={int(inplanes * (2 ** (i + 1)))}')
                layers.append(Bottleneck(type='same', inplanes=int(inplanes * (2 ** (i + 1))), stride=1))
        layers.append(nn.Sequential(nn.BatchNorm2d(1024),
                                    nn.ELU(inplace=True),
                                    ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1024 * 25 * 25 * self.Nslices)
        x = self.fc(x)
        return x


class ClassifierD_l(nn.Module):
    # Based on https://arxiv.org/pdf/1703.09912.pdf
    '''
    Input: layers: length-4 list, default is [3,4,6,3] according to paper
    TODO: HOW to make fc dynamically change the number of input features based on number of slices.
            Right now need to pass it.
    '''

    def __init__(self, Nslices=2):
        super(ClassifierD_l, self).__init__()
        self.net = self._build_block()
        self.Nslices = Nslices
        self.fc = nn.Linear(1024 * 13 * 13 * self.Nslices, 1)

    def _build_block(self, inplanes=1024):
        layers = []
        for j in range(3):
            # print(f'same, inplanes={int(inplanes * (2 ** (i + 1)))}')
            layers.append(Bottleneck(type='same', inplanes=inplanes, stride=1))
        layers.append(Bottleneck(type='quarter', inplanes=inplanes, stride=2))
        for j in range(2):
            # print(f'same, inplanes={int(inplanes * (2 ** (i + 1)))}')
            layers.append(Bottleneck(type='same', inplanes=inplanes, stride=1))
        layers.append(nn.Sequential(nn.BatchNorm2d(inplanes),
                                    nn.ELU(inplace=True),
                                    ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1024 * 13 * 13 * self.Nslices)
        x = self.fc(x)
        return x

class ScaleLayer(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]), requires_grad=True)

    def forward(self, input):
        #print(self.scale)
        return input * self.scale

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
            #self.denoiser =  MRI_UNet(in_channels=1, out_channels=1, f_maps=64,h
            #     layer_order='cr', complex_kernel=False, complex_input=True)

            self.denoiser = UNet(in_channels=1, out_channels=1, f_maps=64, depth=2,
                                 layer_order=['convolution', 'relu'],
                                 complex_kernel=False, complex_input=True,
                                 residual=True, scaled_residual=True)
            print(self.denoiser)
            #self.denoiser = ComplexUNet2D(1, 1, depth=2, final_activation='none', f_maps=32, layer_order='cr')
        elif DENOISER == 'varnet':
            self.varnets = nn.ModuleList()
            for i in range(self.inner_iter):
                self.varnets.append(VarNet())
            self.denoiser = None
        else:
            self.denoiser = CNN_shortcut()
        # self.denoiser = Projector(ENC=False)

    def call_denoiser(self, image):
        image = self.denoiser(image)

        image = self.denoiser(image)
        return(image)

    def set_denoiser_scale(self, scale):
        self.lam2[:] = scale
        self.lam1[:] = 1.0 - self.lam2

    def checkpoint_fn(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, image, kspace, maps, mask):

        for i in range(self.inner_iter):
            # Ex
            #Ex = encoding_op.apply(image)      # For slice by slice:(20, 768, 396, 2) or by case:([slice, 20, 768, 396, 2])
            Ex = sense(maps, image)
            # # kspace correction
            # image_inter = decoding_op.apply(Ex)
            # image_inter_dim = image_inter.ndim
            # if image_inter_dim == 3:
            #     image_inter = torch.unsqueeze(image_inter, 0)
            # image_inter = image_inter.permute(0, -1, 1, 2).contiguous()
            #
            # # Pad to prevent edge effects, circular pad to keep image statistics
            # target_size1 = 32 * math.ceil( (64 + image_inter.shape[-1]) / 32)
            # target_size2 = 32 * math.ceil( (64 + image_inter.shape[-2]) / 32)
            #
            # pad_amount1 = (target_size1 - image_inter.shape[-1]) // 2
            # pad_amount2 = (target_size2 - image_inter.shape[-2]) // 2
            # pad_f = (pad_amount1,pad_amount1,pad_amount2, pad_amount2)
            # image_inter = nn.functional.pad(image_inter,pad_f, "circular")
            # image_inter = self.refiner(image_inter)
            # image_inter = image_inter[:, :, pad_amount2:-pad_amount2, pad_amount1:-pad_amount1]
            # image_inter = image_inter.permute(0, 2, 3, 1).contiguous()
            # if image_inter_dim == 3:
            #     image_inter = torch.squeeze(image_inter)

            # Ex = encoding_op.apply(image_inter)

            # Ex - d
            diff = (Ex - kspace)*mask

            # image = image - scale*E.H*(Ex-d)
            #y_pred = image - self.scale_layers[i] * decoding_op.apply(diff)   # (768, 396)
            image = image - self.scale_layers[i] * sense_adjoint(maps, diff)   # (768, 396)

            # dim = y_pred.ndim
            # if dim == 3:
            #     y_pred = torch.unsqueeze(y_pred,0)
            #     image_complex = torch.unsqueeze(image, 0)
            #     # print(f'image_complex reguires grad {image_complex.requires_grad}')
            # y_pred = y_pred.permute(0, -1, 1, 2).contiguous()
            # image_complex = image_complex.permute(0, -1, 1, 2).contiguous()

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
            # # to complex
            # y_pred = y_pred.permute(0, 2, 3, 1).contiguous()
            # y_pred = torch.view_as_complex(y_pred)
            # y_pred = y_pred[None, ...]
            if self.denoiser is None:
                image_complex = image_complex.permute(0, 2, 3, 1).contiguous()
                image_complex = torch.view_as_complex(image_complex)
                image_complex = image_complex[None, ...]
                # print(f'image_complex reguires grad {image_complex.requires_grad}')
                if self.checkpoint_image:
                    temp = checkpoint.checkpoint(self.checkpoint_fn(self.varnets[i]),image_complex)
                else:
                    temp =self.varnets[i](image_complex)
                image += temp
            else:
                image = image.unsqueeze(0)  #complex64, (1,1,h,w)
                if self.checkpoint_image:
                    image = checkpoint.checkpoint(self.checkpoint_fn(self.denoiser), image)
                else:
                    image = self.denoiser(image)

                # image = self.denoiser(image)
                image = image.squeeze(0)

            if PAD:
                # cropping for UNet
                image = image[:, pad_amount2:-pad_amount2,pad_amount1:-pad_amount1]

            # # back to 2 channel
            # y_pred = y_pred.permute(0, 2, 3, 1).contiguous()
            # y_pred = torch.view_as_real(y_pred[...,0])
            # if dim==3:
            #     y_pred = torch.squeeze(y_pred)
            #
            # #image = self.lam1[i]*image + self.lam2[i]*y_pred
            #
            # # Return Image
            # image = self.lam2[i]*y_pred
        # crop to square
        idxL = int((image.shape[0] - image.shape[1]) / 2)
        idxR = int(idxL + image.shape[1])
        image = image[idxL:idxR,:]

        return image


class unrolledK(nn.Module):
    def __init__(self, scale_init=1.0, inner_iter=1, DENOISER='unet'):
        super(unrolledK, self).__init__()

        self.inner_iter = inner_iter
        self.scale_layers = nn.Parameter(scale_init*torch.ones([inner_iter]), requires_grad=True)

        # Options for UNET
        if DENOISER == 'unet':
            self.denoiser = ComplexUNet2D(1, 1, depth=3, final_activation='none', f_maps=32, layer_order='cr')
        elif DENOISER == 'varnet':
            self.varnets = nn.ModuleList()
            for i in range(self.inner_iter):
                self.varnets.append(VarNet())
            self.denoiser = None

        self.dc_weight = nn.Parameter(torch.ones(1))

    def checkpoint_fn(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, image, kspace, maps, mask):

        k = sense(maps, image)
        for i in range(self.inner_iter):

            zero = torch.zeros(1, 1, 1).to(k)
            dc = torch.where((mask>0), k - kspace, zero) * self.dc_weight

            Ex = sense(maps, image)
            image = torch.unsqueeze(sense_adjoint(maps, Ex), dim=0)
            image = self.denoiser(image)
            image = image[0,...]
            k_update = sense(maps, image)

            k = k - dc - k_update

        return sense_adjoint(maps, k)




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
            # print(f'Target size {target_size1} {target_size2}')

            pad_amount1 = (target_size1 - image.shape[-1]) // 2
            pad_amount2 = (target_size2 - image.shape[-2]) // 2
            # print(f'Target size {pad_amount1} {pad_amount2}')

            # print(f'Target size = {target_size}, pad_amount {pad_amount} input = {y_pred.shape}')

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
            #print(f'image shape {image.shape}')
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


class MoDL_CG(nn.Module):
    # def __init__(self, inner_iter=10):
    def __init__(self, scale_init=1e-3, inner_iter=1):
        super(MoDL_CG, self).__init__()

        # self.encoding_op = encoding_op
        # self.decoding_op = decoding_op

        self.inner_iter = inner_iter
        self.scale_layers = nn.Parameter(torch.ones([inner_iter]), requires_grad=True)
        self.lam = nn.Parameter(torch.ones([inner_iter]), requires_grad=True)
        self.denoiser = DnCNN()

        # nn.ModuleList(self.scale_layers)    # a list of scale layers

    def forward(self, b, encoding_op, decoding_op, encoding_op_torch, decoding_op_torch):
        # Initial guess
        b = torch.squeeze(b)
        image = decoding_op_torch.apply(b)
        # image = torch.zeros([1, 768, 396, 2], dtype=torch.float32)
        # image = image.cuda()

        zeros = torch.zeros(((1,) + image.shape[:-1]), dtype=image.dtype)  # torch.Size([1, 1, 396, 396])
        zeros = zeros.cuda()

        def AhA_I_torch_op(encode, decode, lam):
            AhA = decode * encode
            AhA_I = AhA + lam * sp.linop.Identity(AhA.oshape)
            AhA_I_torch = sp.to_pytorch_function(AhA_I, input_iscomplex=True, output_iscomplex=True)

            return AhA_I_torch

        def grad(b, im, AhA_I_torch, decode_torch):
            # im should be torch tensor ([1, 768, 396, 2])
            return AhA_I_torch.apply(im) - decode_torch.apply(b)

        for i in range(self.inner_iter):
            image = image.permute(0, -1, 1, 2)
            # make images 3 channel.
            image = torch.cat((image, zeros), dim=1)

            # denoised
            #print(image.dtype)
            image = self.denoiser(image)  # torch.Size([1, 3, 396, 396])

            # back to 2 channel
            image = image.permute(0, 2, 3, 1)
            image = image[:, :, :, :-1]

            AhA_I_torch = AhA_I_torch_op(encoding_op, decoding_op, lam=self.scale_layers[i])

            d = - grad(b, image, AhA_I_torch, decoding_op_torch)  # tensor (1, 768, 396, 2)

            Qd = AhA_I_torch.apply(d)
            Qdcpu = d.cpu().numpy()

            dcpu = d.cpu().numpy()
            gH = - np.transpose(np.conj(np.squeeze(chan2_complex(dcpu))))
            gHd = gH @ np.squeeze(chan2_complex(dcpu))  # array (396, 396)

            dHQd = np.transpose(np.conj(np.squeeze(chan2_complex(dcpu)))) @ np.squeeze(
                chan2_complex(Qdcpu))  # array (396, 396)

            alpha = np.sum(-gHd / dHQd)  # should be a number?

            image = image + alpha * d  # should be tensor (1, 768, 396, 2)

        # crop to square here to match ranknet
        idxL = int((image.shape[1] - image.shape[2]) / 2)
        idxR = int(idxL + image.shape[2])
        image = image[:, idxL:idxR, ...]

        # print(image.shape)
        # print('Done')
        return image


# for Bayesian (MSE)
def train_modRecon(
        net: torch.nn.Module,
        train_loader: DataLoader,
        mask_gpu: torch.tensor,
        parameters: Dict[str, float],
        dtype: torch.dtype,
        device: torch.device

) -> nn.Module:
    """
    Train CNN on provided data set.

    Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.

    """
    # Initialize network
    net.to(dtype=dtype, device=device)  # pyre-ignore [28]
    net.train()
    # Define optimizer
    optimizer = optim.Adam(
        net.parameters(),
        lr=parameters.get("lr", 0.001)
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameters.get("step_size", 30)),
        gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
    )
    num_epochs = parameters.get("num_epochs", 1)

    # Train Network
    for _ in range(num_epochs):
        for i, data in enumerate(train_loader,0):

            # get the inputs
            smaps, im, kspaceU = data
            smaps = sp.to_device(smaps, device=spdevice)
            smaps = spdevice.xp.squeeze(smaps)
            smaps = chan2_complex(spdevice.xp.squeeze(smaps))

            kspaceU = sp.to_device(kspaceU, device=spdevice)
            kspaceU = spdevice.xp.squeeze(kspaceU)
            kspaceU = chan2_complex(kspaceU)

            Nslices = smaps.shape[0]
            Ncoils = 20
            xres = 768
            yres = 396

            for sl in range(smaps.shape[0]):
                smaps_sl = smaps[sl]                                # ndarray on cuda (20, 768, 396), complex64
                im_sl = im[sl]                                      # tensor on cuda (768, 396, 2)
                kspaceU_sl = kspaceU[sl]                            # tensor on cuda (20, 768, 396, 2)
                with spdevice:
                    A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
                    Ah = A.H

                A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
                Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)

                imEst = 0.0 * Ah_torch.apply(kspaceU_sl)


                # zero the parameter gradients
                kspaceU_sl = sp.to_pytorch(kspaceU_sl)
                optimizer.zero_grad()
                for inner_iter in range(10):
                # forward + backward + optimize
                    outputs = net(imEst, kspaceU_sl, A_torch, Ah_torch)
                    loss = mseloss_fcn(outputs, im_sl)
                    loss.backward()
                    imEst = outputs
                optimizer.step()
                scheduler.step()

            if i == 9:
                break

    return net


def evaluate_modRecon(
        net: nn.Module, data_loader: DataLoader, mask_gpu: torch.tensor, dtype: torch.dtype, device: torch.device
) -> float:
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # get the inputs
            smaps, im, kspaceU = data
            smaps = sp.to_device(smaps, device=spdevice)
            smaps = spdevice.xp.squeeze(smaps)
            smaps = chan2_complex(spdevice.xp.squeeze(smaps))

            kspaceU = sp.to_device(kspaceU, device=spdevice)
            kspaceU = spdevice.xp.squeeze(kspaceU)
            kspaceU = chan2_complex(kspaceU)

            Nslices = smaps.shape[0]
            Ncoils = 20
            xres = 768
            yres = 396

            for sl in range(Nslices):
                smaps_sl = smaps[sl]  # ndarray on cuda (20, 768, 396), complex64
                im_sl = im[sl] # tensor on cuda (1, 768, 396, 2)
                kspaceU_sl = kspaceU[sl]  # tensor on cuda (20, 768, 396, 2)
                with spdevice:
                    A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
                    Ah = A.H

                A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
                Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)

                imEst = 0.0 * Ah_torch.apply(kspaceU_sl)

                # forward + backward + optimize
                kspaceU_sl = sp.to_pytorch(kspaceU_sl)
                for inner_iter in range(10):
                    outputs = net(imEst, kspaceU_sl, A_torch, Ah_torch)

                    loss = mseloss_fcn(outputs, im_sl)
                    imEst = outputs

            if i == 1:
                break

    return loss
