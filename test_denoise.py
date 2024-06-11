import numpy as np
import cupy as cp
import torch
#import torchvision
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch.linalg
import torchvision.transforms as transforms
import argparse
import matplotlib
matplotlib.use('TKAgg')
import h5py as h5
import pickle
import matplotlib.pyplot as plt
import csv
import logging
import time
import sigpy as sp
import sigpy.mri as mri
from pathlib import Path
import os
import h5py
from IQNet import *
from utils.Recon_helper import *
# from utils.ISOResNet import *
from utils.utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *
from random import randrange
#from fastmri.models.varnet import *


def crop_flipud_tensor(im):
    # expected im shape (1, 768, 396)
    width = im.shape[-1]
    height = im.shape[-2]
    idxL = int((height - width) / 2)
    idxR = int(idxL + width)
    im = im[ :, idxL:idxR, :]
    # flipud
    im = torch.flip(im, dims=(0, 1, 2))
    return im


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_folder = r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\denoise_trained'
val_folder = r'D:\NYUbrain\singleslices\val'
xres = 768
yres = 396
act_xres = 512
act_yres = 256
smap_type = 'smap16'
L2 = True
EFF = False
SUB = True

if EFF:
    if SUB:
        file_model = os.path.join(model_folder, 'DenoiseLoss13351_l2False_effTrue.pt')
        ranknet = EfficientNet2chan()
    else:
        file_model = os.path.join(model_folder, 'DenoiseLoss16976_l2False_effTrue.pt')
        ranknet = EfficientNet2chan()
else:
    file_model = os.path.join(model_folder, 'DenoiseLoss19457_l2True_effFalse.pt')
    ranknet = L2cnn(channels_in=1, channel_base=8, subtract_truth=SUB)

LearnedLoss = Denoise_loss(ranknet)
for param in LearnedLoss.scorenet.parameters():
    param.requires_grad = False
state = torch.load(file_model)
LearnedLoss.load_state_dict(state['state_dict'])
LearnedLoss.cuda()
LearnedLoss.eval()

# Fermi window the data
pady = int(.5 * (yres - act_yres))
padx = int(.5 * (xres - act_xres))
pad = ((padx, xres - padx - act_xres), (pady, yres - pady - act_yres))
[kx, ky] = np.meshgrid(np.linspace(-1, 1, act_xres), np.linspace(-1, 1, act_yres), indexing='ij')
kr = (kx ** 2 + ky ** 2) ** 0.5
mask_truth = (1 / (1 + np.exp((np.abs(kr) - 0.9) / 0.1))).astype(np.float32)
mask_truth = np.pad(mask_truth, pad, 'constant', constant_values=0)
mask_truth = torch.tensor(mask_truth)

# this is the same file used for saliency.py
file = r'D:\NYUbrain\singleslices\val\01409_6_file_brain_AXT2_201_2010549.h5'
with h5py.File(file,'r') as hf:
    smaps = np.array(hf['smaps'])
    smaps = smaps.view(np.int16).astype(np.float32).view(np.complex64)
    smaps /= 32760

    kspace = np.array(hf['kspace'])
    kspace = zero_pad3D(kspace)  # array (sl, coil, 768, 396)
    kspace /= np.max(np.abs(kspace)) / np.prod(kspace.shape[-2:])

kspace_tensor = torch.from_numpy(kspace)
smaps_tensor = torch.from_numpy(smaps)


smaps_tensor = smaps_tensor.cuda()
kspace_tensor = kspace_tensor.cuda()

# truth
im_sl = sense_adjoint(smaps_tensor, kspace_tensor * mask_truth.to(device))
im = crop_flipud_tensor(im_sl)
# normalize
scale = 1.0 / torch.max(torch.abs(im))
im *= scale

im_noisy = add_gaussian2D(im, gaussian_level=4e4, kedge_len=30)      # im_sl torch.Size([1, 396, 396])
im_noisy = im_noisy.unsqueeze(0)
with torch.no_grad():
    imEst = LearnedLoss.denoiser(im_noisy)  # to torch.Size([1, 1, 396, 396])

plt.figure();plt.imshow(np.abs(imEst.squeeze().cpu().numpy()), cmap='gray');plt.show()

out_name = os.path.join(model_folder, 'test_denoise_abs.h5')
with h5py.File(out_name, 'a') as hf:
    # hf.create_dataset(f"truth", data=np.abs(im.squeeze().cpu().numpy()))
    # hf.create_dataset(f"noisy", data=np.abs(im_noisy.squeeze().cpu().numpy()))
    hf.create_dataset(f"L2{L2}_EFF{EFF}_SUB{SUB}", data=np.abs(imEst.squeeze().cpu().numpy()))
