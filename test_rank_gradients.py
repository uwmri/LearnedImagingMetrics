import torch
import torchvision
import torch.optim as optim
import torchsummary
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Ntrial=7685
# Chenweis machine
filepath_rankModel = Path('I:/code/LearnedImagingMetrics_pytorch/Rank_NYU/ImagePairs_Pack_04032020/rank_trained')
filepath_csv = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
rank_channel = 1
rank_trained_on_mag = False
file_rankModel = os.path.join(filepath_rankModel, f"RankClassifier{Ntrial}_pretrained.pt")

ranknet = L2cnn(channels_in=rank_channel)
classifier = Classifier(ranknet)

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False
score = classifier.rank


# hooks for saving forward/gradients
saveForward = saveOutputs()
saveBackward = saveOutputs()
hook_handles = []
hook_handles_backward = []
for layer in score.modules():
    #print(layer)
    # if isinstance(layer, nn.Conv2d):
    handle = layer.register_forward_hook(saveForward)
    handle_backward = layer.register_backward_hook(saveBackward)
    hook_handles.append(handle)
    hook_handles_backward.append(handle_backward)


# load a batch
BATCH_SIZE = 24
NEXAMPLES = 2920
maxMatSize = 396
nch = 2
names = []
files_csv = os.listdir(filepath_csv)
for file in files_csv:
    if fnmatch.fnmatch(file, 'consensus_mode_all.csv'):
        names.append(os.path.join(filepath_csv, file))

# Load the ranks
ranks = []
for fname in names:
    print(fname)
    with open(fname) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            ranks.append(row)
ranks = np.array(ranks, dtype=np.int)
NRANKS = ranks.shape[0]
Labels = np.zeros(NRANKS, dtype=np.int32)
for i in range(0, NRANKS):
    # Label based on ranks from ranker
    if ranks[i, 0] == 2:
        # Same
        Labels[i] = 1
    elif ranks[i, 0] == 1:
        # X_2 is better
        Labels[i] = 0
    else:
        # X_1 is better
        Labels[i] = 2
id = ranks[:,2] - 1

X_1 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize, nch), dtype=np.float32)
X_2 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize, nch), dtype=np.float32)
X_T = np.zeros((NEXAMPLES, maxMatSize, maxMatSize, nch), dtype=np.float32)
filepath_images = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
file ='TRAINING_IMAGES_04032020.h5'
file_images = os.path.join(filepath_images, file)
hf = h5.File(name=file_images, mode='r')
for i in range(NEXAMPLES):

    nameT = 'EXAMPLE_%07d_TRUTH' % (i + 1)
    name1 = 'EXAMPLE_%07d_IMAGE_%04d' % (i + 1, 0)
    name2 = 'EXAMPLE_%07d_IMAGE_%04d' % (i + 1, 1)

    im1 = zero_pad2D(np.array(hf[name1]), maxMatSize, maxMatSize)
    im2 = zero_pad2D(np.array(hf[name2]), maxMatSize, maxMatSize)
    truth = zero_pad2D(np.array(hf[nameT]), maxMatSize, maxMatSize)

    X_1[i] = im1
    X_2[i] = im2
    X_T[i] = truth

    if i % 1e2 == 0:
        print(f'loading example pairs {i + 1}')

X_1 = np.transpose(X_1, [0, 3, 1, 2])
X_2 = np.transpose(X_2, [0, 3, 1, 2])
X_T = np.transpose(X_T, [0, 3, 1, 2])

dataset = DataGenerator_rank(X_1, X_2, X_T, Labels, id, augmentation=True, pad_channels=0)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

torchsummary.summary(score.cpu(), [(X_1.shape[-3], maxMatSize, maxMatSize)
                              ,(X_1.shape[-3], maxMatSize, maxMatSize)], device="cpu")
score.cuda()

for i, data in enumerate(loader, 0):
    # get the inputs
    _, im2, imt, labels = data  # im (batchsize, ch , 396, 396)
    im2, imt = im2.cuda(), imt.cuda()
    im1 = torch.zeros(im2.shape).cuda()
    im1.requires_grad = True
    im2.requires_grad = True
    labels = labels.to(device, dtype=torch.long)

    if i==0:
        score1 = score(im1, imt)
        score1.mean().backward()

        im1_grad = im1.grad
        im1_grad = im1_grad.detach().cpu().numpy()
        im1_gradabs = (im1_grad[:, 0,:,:]**2+im1_grad[:, 1,:,:]**2)**.5

        score2 = score(im2, imt)
        score2.mean().backward()
        im2_grad = im2.grad
        im2_grad = im2_grad.detach().cpu().numpy()
        im2_gradabs = (im2_grad[:, 0, :, :] ** 2 + im2_grad[:, 1, :, :] ** 2) ** .5

        im1_abs = im1.detach().cpu().numpy()
        im1_abs = (im1_abs[:,0,:,:]**2+im1_abs[:,1,:,:]**2)**.5
        im2_abs = im2.detach().cpu().numpy()
        im2_abs = (im2_abs[:,0,:,:]**2+im2_abs[:,1,:,:]**2)**.5
        out_name = os.path.join(filepath_images, f'rank{Ntrial}_im_grad.h5')

        #
        # backward_start = saveBackward.outputs[72][0].cpu().numpy()
        # for m in range(8):
        #     for n in range(4):
        #         plt.subplot(8, 4, (m * 4 + n + 1))
        #         plt.imshow(backward_start[sl, (m * 4 + n), :, :])
        #         plt.axis('off')
        #         plt.colorbar()
        # plt.show()

        break