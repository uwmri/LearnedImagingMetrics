import tkinter as tk
from tkinter import filedialog
import fnmatch
import os
import numpy as np
import cupy
import h5py as h5
import matplotlib.pyplot as plt
import csv
import h5py
import logging

import sigpy as sp
import sigpy.mri as mri
from fastMRI.data import transforms as T

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchsummary
from torch.utils.tensorboard import SummaryWriter

from model_helper import *
from CreateImagePairs import find, get_smaps, get_truth


# def undersample_poisson(ksp_zp, acc=4, Ncoils=20, xres=768, yres=396):
#     """ random undersampling. Input should be (sl,coil, 768, 396) """
#     # logger = logging.getLogger('Poisson Undersampling')
#
#     mask, density = mri.poisson((xres, yres), accel=acc, crop_corner=True, return_density=True)
#     mask = np.broadcast_to(mask, (ksp_zp.shape[0], Ncoils, xres, yres))
#
#     ksp_us = ksp_zp * mask  # ksp2 = (coil, h,w)
#
#     return ksp_us


def get_slices(ksp_zp, Nsl, acc=4):
    """ get N slices from each case, return fully sampled (and undersampled ksp) """
    idx = np.random.randint(0, ksp_zp.shape[0], Nsl)

    # ksp_us = ksp_zp[idx, ...]
    # ksp_us = undersample_poisson(ksp_us, acc=acc, Ncoils=20)

    # return ksp_zp[idx, ...], ksp_us
    return ksp_zp[idx, ...]


class DataGenerator(Dataset):
    def __init__(self, KSP, IMG):
        '''
        :param X_1: X_1_cnnT/V
        :param X_2: X_2_cnnT/V
        :param transform
        '''
        self.KSP = KSP
        self.IMG = IMG
        self.len = KSP.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        kspace = torch.from_numpy(self.KSP[idx,...])
        image = torch.from_numpy(self.IMG[idx,...])

        # if self.transform:

        return kspace, image


# MSE loss
def mseloss_fcn(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss


# learned metrics loss
def learnedloss_fcn(output, target):
    output = output.permute(0,-1,1,2)
    target = target.permute(0, -1, 1, 2)
    delta = (score(output) - score(target)).abs_()
    # loss_fcn = nn.CrossEntropyLoss()
    # loss = loss_fcn(delta, labels)  # labels are 1 (same)

    return delta


class DnCNN(nn.Module):
    def __init__(self, Nlayers=7, Nkernels=64):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(2, Nkernels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(Nlayers - 2):
            layers.append(nn.Sequential(nn.Conv2d(Nkernels, Nkernels, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(Nkernels),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(Nkernels, 2, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        return y - residual



class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale


class MoDL(nn.Module):
    # def __init__(self, inner_iter=10):
    def __init__(self, scale_init=1e-3, inner_iter=10):
        super(MoDL, self).__init__()

        # self.encoding_op = encoding_op
        # self.decoding_op = decoding_op

        self.inner_iter = inner_iter
        self.scale_layers = []
        for i in range(self.inner_iter):
            g = ScaleLayer(init_value=scale_init)
            self.add_module('Scale_%d' % i, g)
            self.scale_layers.append(g)

        nn.ModuleList(self.scale_layers)    # a list of scale layers

        # self.denoiser = DnCNN()

    def forward(self, x, encoding_op, decoding_op, scale_init=1e-3):

        # for i in range(self.inner_iter):
        #     g = ScaleLayer(init_value=scale_init)
        #     self.add_module('Scale_%d' % i, g)
        #     self.scale_layers.append(g)
        # nn.ModuleList(self.scale_layers)

        # Initial guess
        x = np.squeeze(x)
        image = 0.0*decoding_op.apply(x)

        for i in range(self.inner_iter):
            # Gradient descent step

            # Ex
            kspace = encoding_op.apply(image)

            # Ex - d
            kspace -= x

            # x = x  - step*Eh*(Ex-d)
            # image = image - decoding_op.apply(kspace)
            image = image - decoding_op.apply(self.scale_layers[i](kspace))     # (1, 768, 396, 2)

            # reshape to batchSize * channel * h * before feeding into CNN
            image = image.permute(0, -1, 1, 2)  # .contiguous()
            # image = image.unsqueeze(0)

            #Residual Layers
            # image = self.denoiser(image)            # torch.Size([1, 2, 768, 396])
            # print(f'image shape after denoiser {image.shape}')

            # image = image.squeeze(0)
            image = image.permute(0,2,3,1)#.contiguous()

        # crop to square here to match ranknet
        idxL = int((image.shape[1] - image.shape[2]) / 2)
        idxR = int(idxL + image.shape[2])
        image = image[:, idxL:idxR, ...]
        # print(image.shape)
        #print('Done')
        return image


class AutoMap(nn.Module):
    def __init__(self, ishape, oshape):
        super(AutoMap, self).__init__()
        self.ishape = ishape    # (batchSize, 2, coil, xres, yres)
        self.oshape = oshape    # (batchSize, 2, xres, yres)
        self.xres = ishape[-2]
        self.yres = ishape[-1]

        self.ireshape = int(np.prod(self.ishape))
        self.oreshape = int(np.prod(self.oshape))

        self.domain_transform = nn.Linear(self.ireshape, 2*self.oreshape)
        self.domain_transform2 = nn.Linear(2*self.oreshape, self.oreshape)

        # TO DO: add initialize fc as a FT

        self.encoder = nn.Sequential(nn.Conv2d(ishape[1], 64, 5, 1, 2), nn.ReLU(True),
                             nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(True),
                             nn.ConvTranspose2d(64, oshape[1], 7, 1, 3))


    def forward(self, x):

        """ input kspace should be (batchSize, channel, coil, h, w) """

        batchSize = len(x)
        x = x.reshape(batchSize, int(np.prod(self.ishape)))
        x = F.tanh(self.domain_transform(x))
        x = F.tanh(self.domain_transform2(x))
        x = x.reshape(-1, *self.oshape)
        x = self.encoder(x)

        return x



mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()

device = sp.Device(0)

# load RankNet
root = tk.Tk()
root.withdraw()
filepath_rankModel = tk.filedialog.askdirectory(title='Choose where the saved metric model is')
file_rankModel = os.path.join(filepath_rankModel, "RankClassifier.pt")
classifier = Classifier()

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
score = classifier.rank
score.cuda()

# Get kspace data. (sl, coil, 2, kx, ky)
root = tk.Tk()
root.withdraw()
filepath_raw = tk.filedialog.askdirectory(title='Choose where the raw file is')
files_raw = find("*.h5", filepath_raw )

# Only choose 20-coil data for now
Ncoils = 20
xres = 768
yres = 396
acc = 4

ksp_full = []
ksp_us = []
index_file = 0

# fixed sampling mask
mask, density = mri.poisson((xres, yres), accel=acc, crop_corner=True, return_density=True)



while index_file < 2:

    file = files_raw[np.random.randint(len(os.listdir(filepath_raw)))]
    print(f'reading kspace from {file}')

    hf = h5py.File(file, mode='r')

    ksp_temp = hf['kspace'][()]
    if ksp_temp.shape[1] == Ncoils:

        print(f'loading kspace from {file}')
        ksp_temp = zero_pad4D(ksp_temp)

        # get N slices per case
        ksp_temp = get_slices(ksp_temp, Nsl=1, acc=4)       # (sl, coil, h, w)

        # undersample
        mask = np.broadcast_to(mask, (ksp_temp.shape[0], Ncoils, xres, yres))
        ksp_temp_us = ksp_temp * mask

        ksp_full.append(ksp_temp)
        ksp_us.append(ksp_temp_us)

        index_file += 1
    else:
        print(f'Not enough coils, pass')

ksp_full = np.asarray(ksp_full)
ksp_us = np.asarray(ksp_us)
ksp_full = np.reshape(ksp_full, (-1,) + ksp_full.shape[2:])
ksp_us = np.reshape(ksp_us, (-1,) + ksp_us.shape[2:])   #(total slices, 768, 396)
ksp_us = ksp_us.astype('complex64')

# Get smaps
smaps = np.zeros(ksp_full.shape, dtype=ksp_full.dtype)
for sl in range(ksp_full.shape[0]):
    ksp_sl_gpu = sp.to_device(ksp_full[sl], device=device)
    mps = get_smaps(ksp_sl_gpu, device=device, maxiter=50)
    smaps[sl] = sp.to_device(mps, sp.cpu_device)

    del mps, ksp_sl_gpu
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

# # flip to be the same orientation as image_truth
# smaps_upright = smaps[:,:,::-1,:]

# Get True images
# (Nsl, 768, 396)
image_truth = np.zeros((ksp_full.shape[0],ksp_full.shape[-2], ksp_full.shape[-1]),dtype=ksp_full.dtype)
for sl in range(ksp_full.shape[0]):
    mps = sp.to_device(smaps[sl], device)
    image_truth[sl] = get_truth(ksp_full, sl, device=device, lamda=0.005, smaps=mps, forRecon=True)

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

# Generate Batched training/validation set
NEXAMPLES = image_truth.shape[0]

image_truth_stack = np.stack((np.real(image_truth), np.imag(image_truth)), axis=-1)
ksp_us_stack = np.stack((np.real(ksp_us), np.imag(ksp_us)), axis=-1)
# print(image_truth_stack.shape)  # (sl, 396, 396, 2)
# print(ksp_us_stack.shape)       # (sl, coil, 768, 396, 2)

ntotal = NEXAMPLES
ntrain = int(0.9*NEXAMPLES)

image_truthT = image_truth_stack[:ntrain,...]
ksp_usT = ksp_us_stack[:ntrain,...]
image_truthV = image_truth_stack[ntrain:,...]
ksp_usV = ksp_us_stack[ntrain:,...]

# torch tensor should be minibatch * channel * H*W
# image_truthT = np.transpose(image_truthT,[0,3,1,2])
# ksp_usT = np.transpose(ksp_usT,[0,4,1,2,3])
# image_truthV = np.transpose(image_truthV,[0,3,1,2])
# ksp_usV = np.transpose(ksp_usV,[0,4,1,2,3])


# crop truth
idxL = int((image_truthV.shape[1] - image_truthV.shape[2]) / 2)
idxR = int(idxL + image_truthV.shape[2])
image_truthV = image_truthV[:, idxL:idxR, ...]
image_truthT = image_truthT[:, idxL:idxR, ...]
print(f'Training set image_truth size {image_truthT.shape}')
print(f'Training set ksp_us size {ksp_usT.shape}')


# Data generator
BATCH_SIZE = 1
trainingset = DataGenerator(image_truthT,ksp_usT)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

validationset = DataGenerator(image_truthV,ksp_usV)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=True)

# test undersampling, SOS recon
show_sos = False
if show_sos:
    sl = np.random.randint(ksp_full.shape[0])
    ksp2 = ksp_us[sl]

    ksp2_tensor = T.to_tensor(ksp2)  # T requires array to be tensor
    image = T.ifft2(ksp2_tensor)  # torch.Size([coil, 640, 320, 2])
    image = image.numpy()
    image = image[:, :, :, 0] + 1j * image[:, :, :, 1]
    # abs before square. There will be shading if don't do this.
    image = np.abs(image)
    image = np.sqrt(np.sum(image ** 2, axis=0))

    # crop and flipud
    width = image.shape[1]
    lower = int(.5 * width)
    upper = int(1.5 * width)
    image_sq = image[lower:upper, :]
    image_sq = np.flipud(image_sq)

    plt.imshow(image_sq, cmap='gray')
    plt.show()


imSlShape = (BATCH_SIZE,) + smaps.shape[-2:]

UNROLL = True
if UNROLL:


        # Setup Network
        ReconModel = MoDL()


else:   # AUTOMAP
    # torch tensor should be minibatch * channel * H*W
    image_truthT = np.transpose(image_truthT,[0,3,1,2])
    ksp_usT = np.transpose(ksp_usT,[0,4,1,2,3])
    image_truthV = np.transpose(image_truthV,[0,3,1,2])
    ksp_usV = np.transpose(ksp_usV,[0,4,1,2,3])

    BATCH_SIZE = 1
    trainingset = DataGenerator(image_truthT, ksp_usT)
    loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

    validationset = DataGenerator(image_truthV, ksp_usV)
    loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=True)

    ksp_batchShape = (BATCH_SIZE, ) + ksp_usT.shape[1:]
    im_batchShape = (BATCH_SIZE, ) + image_truthT.shape[1:]
    ReconModel = AutoMap(ishape= ksp_batchShape, oshape=im_batchShape)

ReconModel.cuda()
optimizer = torch.optim.Adam(ReconModel.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                             amsgrad=False)


# training
Ntrial = 1
writer_train = SummaryWriter(f'runs/recon/train_{Ntrial}')
writer_val = SummaryWriter(f'runs/recon/val_{Ntrial}')

WHICH_LOSS = 'mse'

Nepoch = 10
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)

for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    ReconModel.train()

    for i, data in enumerate(loader_T, 0):
        im, kspaceU = data

        # print(f'{kspaceU.shape}')   # torch.Size([1, 20, 768, 396, 2])
        # print(f'{im.shape}')        # torch.Size([1, 768, 396, 2])

        im, kspaceU = im.cuda(), kspaceU.cuda()

        smaps_sl = sp.to_device(smaps[i], device=device)
        with device:
            A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, ishape=imSlShape)
            Ah = A.H

        AhA = Ah * A
        max_eigen = sp.app.MaxEig(AhA, dtype=smaps_sl.dtype, device=device, max_iter=30).run()

        A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
        Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)

        # forward
        # imEst = ReconModel(kspaceU, A_torch, Ah_torch, scale_init=0.9 / max_eigen)
        imEst = ReconModel(kspaceU, A_torch, Ah_torch, scale_init=0.001/max_eigen)
        # loss
        if WHICH_LOSS == 'mse':
            loss = mseloss_fcn(imEst, im)
        else:
            # label = torch.tensor(np.ones(BATCH_SIZE, dtype='int32'))
            # loss = learnedloss_fcn(imEst, im, label)
            loss = learnedloss_fcn(imEst, im)

        train_avg.update(loss.item(), n=BATCH_SIZE)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Validation
    with torch.no_grad():
        for i, data in enumerate(loader_V, 0):
            im, kspaceU = data
            im, kspaceU = im.cuda(), kspaceU.cuda()

            # forward
            imEst = ReconModel(kspaceU, A_torch, Ah_torch, scale_init=0.9 / max_eigen)

            if WHICH_LOSS == 'mse':
                loss = mseloss_fcn(imEst, im)
            else:
                loss = learnedloss_fcn(imEst, im)
            eval_avg.update(loss.item(), n=BATCH_SIZE)

    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)

    print('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
    lossT[epoch] = train_avg.avg()
    lossV[epoch] = eval_avg.avg()


