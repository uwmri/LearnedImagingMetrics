import numpy as np
import sigpy as sp
import os
import h5py
import torch
import torchvision
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.utils_augmentation import add_phase_im, sigpy_image_rotate2
from utils.utils import zero_pad3D

def acc_calc(output, labels, BatchSize=16):
    '''
    Returns accuracy of a mini-batch

    '''

    _, preds = torch.max(output, 1)

    return (preds == labels).sum().item()/BatchSize


class RunningAcc:
    def __init__(self):  # initialization
        self.count = 0
        self.sum = 0

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value       # sum N batches acc

    def avg(self):
        return self.sum / self.count    # avg acc over batches


class RunningAverage:
    def __init__(self):  # initialization
        self.count = 0
        self.sum = 0

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n

    def avg(self):
        return self.sum / self.count


def get_class_weights(labels):
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)], dtype='float32')
    weights = class_sample_count / class_sample_count.max()
    weights = torch.from_numpy(weights)

    return weights.cuda()

######################################### Loss functions ###############################################################
def mseloss_fcn(output, target):
    mse = torch.abs(output - target)
    mse = mse.view(mse.shape[0], -1)
    mse = torch.sum(mse ** 2, dim=1, keepdim=True) ** 0.5

    return mse


def mseloss_fcn0(output, target):
    loss = torch.mean(torch.abs(output - target) ** 2) ** 0.5
    return loss


def learnedloss_fcn(output, target, scoreModel, rank_trained_on_mag=False, augmentation=False):

    if output.ndim == 2:
        output = output.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    if output.ndim == 3:
        output = torch.unsqueeze(output, 0)
        target = torch.unsqueeze(target, 0)
    Nslice = output.shape[0]
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


########################################################################################################################


################################################### DataGenerators #####################################################

class DataGenerator_rank(Dataset):
    def __init__(self, X_1, X_2, X_T, Y, ID, augmentation=False,  roll_magL=-15, roll_magH=15,
                 crop_sizeL=1, crop_sizeH=15, scale_min=0.2, scale_max=2.0, kshift_max=10,device=sp.Device(0), pad_channels=0):

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
        self.kshift_max = kshift_max

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, idx):
        xp = self.device.xp

        if self.augmentation:
            FLIP = np.ndarray.item(np.random.choice([False, True], size=1, p=[0.5, 0.5]))
            #scale = np.random.random()*(self.scale_max-self.scale_min) + self.scale_min
            scale = 1.0
            ROT = True
            ROLL = True
            LPHASE = True
            CONSTPHASE = True
        else:
            FLIP = False
            ROT = False
            ROLL = False
            LPHASE = False
            CONSTPHASE = False
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

        # Add linear phase to the 'bad' image
        if LPHASE:
            kshift = np.random.randint(-self.kshift_max, self.kshift_max)
            x1 = add_phase_im(x1, kshift=kshift)
            x2 = add_phase_im(x2, kshift=kshift)
            xt = add_phase_im(xt, kshift=kshift)

        if CONSTPHASE:
            phi0 = (2*np.random.random_sample()-1) * np.pi
            x1 = x1 * phi0
            x2 = x2 * phi0
            xt = xt * phi0

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

        y = self.Y[idx]

        return x1, x2, xt, y


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
            else:
                smaps = np.array(hf['smaps'])

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


######################################### SSIM and MSE modules #########################################################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
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


class MSEmodule(nn.Module):
    def __init__(self):
        super(MSEmodule, self).__init__()

    def forward(self, x, truth):
        y = torch.abs(x - truth)
        y = y.view(y.shape[0], -1)
        return torch.sum(y**2, dim=1, keepdim=True)**0.5


