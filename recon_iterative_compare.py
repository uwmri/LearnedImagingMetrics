import torch
import torchvision
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch.linalg

import cupy
import h5py as h5
import pickle

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import csv
import logging
import time
import sigpy as sp
import sigpy.mri as mri
from pathlib import Path
import os
import h5py
import pandas as pd

from utils.Recon_helper import *
from utils.model_helper import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_slices', type=int, default=1)
parser.add_argument('--project_dir', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch')
parser.add_argument('--data_dir', type=str,
                    default=r'I:\NYUbrain\singleslices\val\T1')
parser.add_argument('--output_dir', type=str,
                    default=r'I:\NYUbrain\poisson_acc8x2')
args = parser.parse_args()

xres = 768
yres = 396
act_xres = 512
act_yres = 256
acc = 8
masks = []
for m in range(args.num_slices):
    mask = mri.poisson((act_xres, act_yres), accel=acc *2 , calib=(0, 0), crop_corner=True, return_density=False,
                   dtype='float32')
    pady = int(.5 * (yres - act_yres))
    padx = int(.5 * (xres - act_xres))
    # print(mask.shape)
    print(f'padx = {(padx, xres - padx - act_xres)}, {(pady, yres - pady - act_yres)}')
    pad = ((padx, xres - padx - act_xres), (pady, yres - pady - act_yres))
    mask = np.pad(mask, pad, 'constant', constant_values=0)
    masks.append(mask)

# Fermi window the data
[kx, ky] = np.meshgrid(np.linspace(-1, 1, act_xres), np.linspace(-1, 1, act_yres), indexing='ij')
# [kx, ky] = np.meshgrid(np.linspace(-1, 1, xres), np.linspace(-1, 1, yres), indexing='ij')
kr = (kx ** 2 + ky ** 2) ** 0.5
mask_truth = (1 / (1 + np.exp((np.abs(kr) - 0.9) / 0.1))).astype(np.float32)
mask_truth = np.pad(mask_truth, pad, 'constant', constant_values=0)
# mask_truth = torch.ones((xres, yres))


effective_acc = np.count_nonzero(masks[0]) / np.count_nonzero(mask_truth)
print(f'effective acc {effective_acc}')

contrasts = ['T1']
fname = '01275_0_file_brain_AXT1_201_6002725.h5'
with h5py.File(os.path.join(args.data_dir, fname), 'r') as hf:
    # for smaps16
    smaps = np.array(hf['smaps'])
    smaps = smaps.view(np.int16).astype(np.float32).view(np.complex64)
    smaps /= 32760
    kspace = np.array(hf['kspace'])
kspace = zero_pad3D(kspace)  # array (sl, coil, 768, 396)
kspace /= np.max(np.abs(kspace))/np.prod(kspace.shape[-2:])

mask_gpu = sp.to_device(masks, sp.Device(0))


kspaceU_gpu = sp.to_device(kspace, sp.Device(0)) * mask_gpu
smaps_gpu = sp.to_device(smaps, sp.Device(0))

width = kspace.shape[2]
height = kspace.shape[1]
idxL = int((height - width) / 2)
idxR = int(idxL + width)


imSense = sp.mri.app.SenseRecon(kspaceU_gpu, smaps_gpu, weights=mask_gpu, lamda=.01,
                                                            max_iter=30, device=spdevice).run()
imSense = imSense[idxL:idxR, :]
# L1-wavelet
imL1W = sp.mri.app.L1WaveletRecon(kspaceU_gpu, smaps_gpu, weights=mask_gpu, lamda=.001,
                                 max_iter=30, device=spdevice).run()
imL1W = imL1W[idxL:idxR, :]

# Get zerofilled image to estimate max
imU_sl = sense_adjoint(sp.to_pytorch(smaps_gpu), sp.to_pytorch(kspaceU_gpu) * sp.to_pytorch(mask_truth).cuda())
imU_sl = imU_sl.squeeze()
imU_sl = imU_sl[idxL:idxR, :]

with h5py.File(os.path.join(args.output_dir, 'ReconIter_T1.h5'), 'a') as hf:
    hf.create_dataset(f"SENSE_mag", data=np.abs(np.squeeze(imSense.get())))
    hf.create_dataset(f"SENSE_phase", data=np.angle(np.squeeze(imSense.get())))
    hf.create_dataset(f"L1Wavelet_mag", data=np.abs(np.squeeze(imL1W.get())))
    hf.create_dataset(f"L1Wavelet_phase", data=np.angle(np.squeeze(imL1W.get())))