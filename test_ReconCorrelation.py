from scipy import ndimage

import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.Recon_helper import *
from utils.ISOResNet import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *
from utils.utils import *

import torchvision.transforms.functional as TF
import sigpy as sp
import math
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import h5py as h5
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Ntrial = 5888
filepath_rankModel = Path(r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn')
file_rankModel = os.path.join(filepath_rankModel, f"RankClassifier{Ntrial}.pt")

rank_channel =1
ranknet = L2cnn(channels_in=rank_channel, channel_base=8, train_on_mag=False)
classifier = Classifier(ranknet)

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
scoreModel = classifier.rank
# for param in classifier.parameters():
#     param.requires_grad = False
scoreModel.cuda()

"""
    the same image every 20 epochs
"""
file_images = r'Y:\ctang\test_score_mse\Images_training1827_learned.h5'
hf = h5.File(name=file_images, mode='r')

images = []
keys = [f for f in hf.keys()]
for i in range(len(keys)):
    if bool(re.match(r".*_recon_im",keys[i])):
        image_im = np.array(hf[keys[i]])
        if bool(re.match(keys[i][:-2]+r"re",keys[i+1])):
            image_re = np.array(hf[keys[i+1]])
            image = image_re + 1j* image_im
            #print(keys[i+1])
            images.append(image)
images = np.array(images)

for i in range(10):
    if bool(re.match(r"0_truth_im",keys[i])):
        truth_im = np.array(hf[keys[i]])
        if bool(re.match(r"0_truth_re",keys[i+1])):
            truth_re = np.array(hf[keys[i+1]])
            truth = truth_re + 1j* truth_im
truth = torch.from_numpy(truth).cuda()
truth = torch.unsqueeze(truth, 0)
truth = torch.unsqueeze(truth, 0)
truth_combined = torch.cat((truth, truth), 0)

scorelist=[]
scorelist1=[]
mselist = []
for i in range(images.shape[0]):
    im = torch.from_numpy(images[i]).cuda()
    im = torch.unsqueeze(im, 0)
    im = torch.unsqueeze(im, 0)
    im_combined= torch.cat((im, im), dim=0)
    scores = scoreModel(im_combined, truth_combined).detach().cpu().numpy().squeeze()
    scorelist.append(scores[0])
    scorelist1.append(scores[1])
    mse = torch.abs(im - truth)
    mse = mse.view(mse.shape[0], -1)
    mse = torch.sum(mse**2, dim=1, keepdim=True)**0.5
    #mse = torch.mean(torch.abs(im - truth) ** 2) ** 0.5
    mselist.append(mse.item())

scorelist = np.array(scorelist)
mselist = np.array(mselist)
plt_scoreVsMse(scorelist, mselist)
plt.show()

"""
    All images at first epoch for recon training
"""
file_images_recon = r'I:\code\LearnedImagingMetrics_pytorch\ReconTraining_4294.h5'
hf2 = h5.File(name=file_images_recon, mode='r')
imEst2 = []
im_sl = []
keys = [f for f in hf2.keys()]
for i in range(len(keys)):
    if bool(re.match(r".case.*",keys[i])):
        imEst2.append(np.array(hf2[keys[i]]))
    elif bool(re.match(r"truth.*case.*",keys[i])):
        im_sl.append(np.array(hf2[keys[i]]))
imEst2 = np.array(imEst2)
im_sl = np.array(im_sl)
imEst2_tensor = torch.from_numpy(imEst2).cuda()
imEst2_tensor = torch.unsqueeze(imEst2_tensor, 1)
im_sl_tensor = torch.from_numpy(im_sl).cuda()
im_sl_tensor = torch.unsqueeze(im_sl_tensor, 1)

scores = scoreModel(imEst2_tensor, im_sl_tensor).detach().cpu().numpy().squeeze()
mse = torch.abs(imEst2_tensor - im_sl_tensor)
mse = mse.view(mse.shape[0], -1)
mse = torch.sum(mse**2, dim=1, keepdim=True)**0.5

plt_scoreVsMse(scores, mse.cpu().numpy().squeeze())
plt.show()

