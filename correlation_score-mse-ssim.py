import logging

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity
from random import randrange
import sigpy.mri as mri
import pywt
import time
import matplotlib
matplotlib.use('TKAgg')
import scipy
import argparse
import h5py
from pathlib import Path


from utils.utils import *
from utils.CreateImagePairs import *
from utils.ISOResNet import *
from utils.utils_DL import *

spdevice = sp.Device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--file_images', type=str, default=Path(r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\TRAINING_IMAGES_04032020.h5'))
args = parser.parse_args()

filepath_rankModel = Path(r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn\CV-5')
train_folder = Path("D:/NYUbrain/brain_multicoil_train/multicoil_train")
val_folder = Path("D:/NYUbrain/brain_multicoil_val/multicoil_val")
test_folder = Path("D:/NYUbrain/brain_multicoil_test/multicoil_test")
files = find("*.h5", train_folder)

with open(r"I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\file.txt", "r") as log_file:
    log = log_file.readlines()
NEXAMPLES = len(log)
# NEXAMPLES = 3
scorelist = []
mselist = []
ssimlist = []
for ss in range(NEXAMPLES):
    print(f'{ss}/{NEXAMPLES}')
    file = log[ss].split(' ')[4]
    hf_raw = h5py.File(file, mode='r')

    ksp = hf_raw['kspace'][()]
    tot_slices = ksp.shape[0]
    num_slice = int(log[ss].split(' ')[2])
    if num_slice > np.floor(tot_slices * 0.75):
        pass
    else:
        print(f'{num_slice}')
        count = int(log[ss].split(' ')[-1])
        with h5py.File(name=args.file_images, mode='r') as hf:
            # Just read and subtract truth for now, index later
            nameT = 'EXAMPLE_%07d_TRUTH' % (count )
            name1 = 'EXAMPLE_%07d_IMAGE_%04d' % (count , 0)
            name2 = 'EXAMPLE_%07d_IMAGE_%04d' % (count, 1)

            im1 = np.array(hf[name1]).astype('complex64')
            im2 = np.array(hf[name2]).astype('complex64')
            imT = np.array(hf[nameT]).astype('complex64')


            im1_tensor = torch.unsqueeze(torch.from_numpy(im1.copy()), 0)
            im1_tensor = im1_tensor.unsqueeze(0)
            im2_tensor = torch.unsqueeze(torch.from_numpy(im2.copy()), 0)
            im2_tensor = im2_tensor.unsqueeze(0)
            image_truth_tensor = torch.unsqueeze(torch.from_numpy(imT.copy()), 0)
            image_truth_tensor = image_truth_tensor.unsqueeze(0)
            image_truth_tensor, im1_tensor, im2_tensor = image_truth_tensor.cuda(), im1_tensor.cuda(), im2_tensor.cuda()

            scores1 = []
            scores2 = []
            metric_files = os.listdir(filepath_rankModel)
            for i_cv in range(5):

                file_rankModel = os.path.join(filepath_rankModel, metric_files[i_cv])
                log_dir = filepath_rankModel

                rank_channel = 1
                # ranknet = ISOResNet2(BasicBlock, [2,2,2,2], for_denoise=False)
                ranknet = L2cnn(channels_in=rank_channel, channel_base=8)
                classifier = Classifier(ranknet)

                state = torch.load(file_rankModel)
                classifier.load_state_dict(state['state_dict'], strict=True)
                classifier.eval()
                for param in classifier.parameters():
                    param.requires_grad = False
                scoreNet = classifier.rank
                scoreNet.cuda()
                score1 = scoreNet(im1_tensor, image_truth_tensor)
                score2 = scoreNet(im2_tensor, image_truth_tensor)

                scores1.append(score1.squeeze().cpu().numpy())
                scores2.append(score2.squeeze().cpu().numpy())

            scores1 = np.array(scores1)
            scoreMean1 = np.mean(scores1)
            scores2 = np.array(scores2)
            scoreMean2 = np.mean(scores2)

            mse1 = np.sum((np.abs(im1) - np.abs(imT)) ** 2) ** 0.5
            mse2 = np.sum((np.abs(im2) - np.abs(imT)) ** 2) ** 0.5
            ssim1 = structural_similarity(np.abs(im1), np.abs(imT))
            ssim2 = structural_similarity(np.abs(im2), np.abs(imT))


            scorelist.append(scoreMean1)
            scorelist.append(scoreMean2)
            mselist.append(mse1)
            mselist.append(mse2)
            ssimlist.append(ssim1)
            ssimlist.append(ssim2)

scorelist = np.array(scorelist)
mselist = np.array(mselist)
ssimlist = np.array(ssimlist)

from scipy import stats
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(mselist, scorelist)
slope1, intercept1, r_value1, p_value1, slope_std_error1 = stats.linregress((1-ssimlist), scorelist)
print(f'MSE-Score: y = {slope} * x + {intercept}, p={p_value}, r2={r_value}, sigma={slope_std_error}')
print(f'SSIM-Score: y = {slope1} * x + {intercept1}, p={p_value1}, r2={r_value1}, sigma={slope_std_error1}')

import pandas as pd
dataframe=pd.DataFrame(data=[mselist, (1-ssimlist), scorelist]).T
dataframe.columns=['MSE','1-SSIM','Score']
import seaborn

fig_ssim, ax1 = plt.subplots(figsize=(7, 5))
ax1.set_xlabel('1-ssim', fontsize=28)
ax1.set_ylabel('Score', fontsize=26)
seaborn.regplot(x="1-SSIM", y="Score", fit_reg=False, data=dataframe,  scatter_kws={'alpha':0.07, 's':30});
ax1.tick_params(axis='y')
ax1.tick_params(axis='both', labelsize=22)
fig_ssim.tight_layout()  # otherwise the right y-label is slightly clipped
fig_ssim.savefig(f'cv5Mean_correlation_ssim-score.png')

fig_mse, ax = plt.subplots(figsize=(7, 5))
ax.set_xlabel('MSE', fontsize=28)
ax.set_ylabel('Score',  fontsize=26)
seaborn.regplot(x="MSE", y="Score", fit_reg=False, data=dataframe, scatter_kws={'alpha':0.07, 's':30})
ax.tick_params(axis='y')
ax.tick_params(axis='both', labelsize=22)
fig_mse.tight_layout()  # otherwise the right y-label is slightly clipped
fig_mse.savefig(f'cv5Mean_correlation_mse-score.png')

# plt.title('Scatterplot for the Association between Breast Cancer and Internet Usage');

