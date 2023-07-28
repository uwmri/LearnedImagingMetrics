import torch
import torchvision
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch.linalg

import cupy
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

from utils.Recon_helper import *
from utils.ISOResNet import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *
from random import randrange
#from fastmri.models.varnet import *

class DataGeneratorConvert(Dataset):

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
        smaps = np.array(self.hf[self.scans[actual_idx]]['smaps'])
        kspace = np.array(self.hf[self.scans[actual_idx]]['kspace'])

        if not self.case_name:
            return smaps, kspace
        else:
            return smaps, kspace, self.scans[actual_idx]




file_train = 'ksp_truths_smaps_train_lzf.h5'
file_val = 'ksp_truths_smaps_val_lzf.h5'
smap_type = 'smap16'

# Only choose 20-coil data for now
Ncoils = 20
xres = 768
yres = 396
act_xres = 512
act_yres = 256

data_folder= 'E:\\LearnedImageMetric\\'
out_folder = 'Q:\\FastMRI\\'

trainingset = DataGeneratorConvert(data_folder, file_train, rank_trained_on_mag=False, data_type=smap_type, case_name=True)
validationset = DataGeneratorConvert(data_folder, file_val, rank_trained_on_mag=False, data_type=smap_type, case_name=True)

os.mkdir(os.path.join(out_folder, 'train'))
count = 0
for smaps, kspace, name in trainingset:
    print(f'{count} of {len(trainingset)} : {name}')

    for slice in range(smaps.shape[0]):
        fname = f'{count:05d}_{slice}_{name}'
        out_name = os.path.join(out_folder,'train',fname)

        # Export the data
        with h5py.File(out_name, 'w') as hf:
            hf.create_dataset('smaps', data=smaps[slice], compression="lzf")
            hf.create_dataset('kspace', data=kspace[slice], compression="lzf")

    count = count + 1


os.mkdir(os.path.join(out_folder, 'val'))
for smaps, kspace, name in validationset:
    print(f'{count-len(trainingset)} of {len(validationset)} : {name}')

    for slice in range(smaps.shape[0]):
        fname = f'{count:05d}_{slice}_{name}'
        out_name = os.path.join(out_folder,'val',fname)

        # Export the data
        with h5py.File(out_name, 'w') as hf:
            hf.create_dataset('smaps', data=smaps[slice], compression="lzf")
            hf.create_dataset('kspace', data=kspace[slice], compression="lzf")

    count = count + 1