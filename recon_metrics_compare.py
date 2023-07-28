"""
This file grabs N slices, do the recon with MoDL trained with IQNet, MSE and SSIM
and compares the reconstructed image IQNet score, MSE and SSIM.
This is the script that generated the boxplots in the paper.
"""
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
import pandas as pd
from IQNet import *
from utils.Recon_helper import *
from utils.ISOResNet import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_score(idx, imEstL, im_sl, scorenets, scorelist, rank_trained_on_mag=False, augmentation=False):
    learnedlossL = 0.0
    for score in scorenets:
        learnedlossL += learnedloss_fcn(imEstL, im_sl, score, rank_trained_on_mag=rank_trained_on_mag, augmentation=augmentation)
    learnedlossL /= 5
    scorelist[idx] = learnedlossL.cpu().numpy()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_slices', type=int, default=5)
parser.add_argument('--reconID_learned', type=str, default='1980')
parser.add_argument('--reconID_mse', type=str, default='8470')
parser.add_argument('--reconID_ssim', type=str, default='3987')
parser.add_argument('--recon_dir', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\recon_models')
parser.add_argument('--rank_dir', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn\CV-5')
parser.add_argument('--project_dir', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch')
parser.add_argument('--data_dir', type=str,
                    default=r'I:\NYUbrain\singleslices\val')
args = parser.parse_args()



recon_file = os.path.join(args.recon_dir,f'Recon{args.reconID_learned}_learned.pt')
state = torch.load(recon_file)
ReconModel = MoDL(inner_iter=5, DENOISER='unet')
ReconModel.load_state_dict(state['state_dict'], strict=True)
ReconModel.cuda();
ReconModel.eval()

recon_fileMSE = os.path.join(args.recon_dir, f'Recon{args.reconID_mse}_mse.pt')
stateMSE = torch.load(recon_fileMSE)
ReconModelMSE = MoDL(inner_iter=5, DENOISER='unet')
ReconModelMSE.load_state_dict(stateMSE['state_dict'], strict=True)
ReconModelMSE.cuda();
ReconModelMSE.eval()

recon_fileSSIM = os.path.join(args.recon_dir, f'Recon{args.reconID_ssim}_ssim.pt')
stateSSIM = torch.load(recon_fileSSIM)
ReconModelSSIM = MoDL(inner_iter=5, DENOISER='unet')
ReconModelSSIM.load_state_dict(stateSSIM['state_dict'], strict=True)
ReconModelSSIM.cuda();
ReconModelSSIM.eval()

metric_files =  [f for f in os.listdir(args.rank_dir) if os.path.isfile(os.path.join(args.rank_dir, f))]
scorenets = []
for name in metric_files:
    ranknet = L2cnn(channels_in=1, channel_base=8, group_depth=5, train_on_mag=False)

    classifier = Classifier(ranknet)

    state = torch.load(os.path.join(args.rank_dir, name))
    classifier.load_state_dict(state['state_dict'], strict=True)
    classifier.eval()
    score = classifier.rank
    score.to(device)

    scorenets.append(score)

xres = 768
yres = 396
act_xres = 512
act_yres = 256
acc = 8
masks = []
for m in range(args.num_slices):
    # mask = mri.poisson((act_xres, act_yres), accel=acc *2 , calib=(0, 0), crop_corner=True, return_density=False,
    #                    dtype='float32')
    mask = np.ones((act_xres, act_yres), dtype='float32')
    pady = int(.5 * (yres - act_yres))
    padx = int(.5 * (xres - act_xres))
    # print(mask.shape)
    print(f'padx = {(padx, xres - padx - act_xres)}, {(pady, yres - pady - act_yres)}')
    pad = ((padx, xres - padx - act_xres), (pady, yres - pady - act_yres))
    mask = np.pad(mask, pad, 'constant', constant_values=0)
    mask = mri.poisson((xres, yres), accel=acc, calib=(0, 0), crop_corner=True, return_density=False,
                       dtype='float32')

    # random PE lines
    # calib = 24
    # num_peri = int((xres * yres / acc - xres * calib) / xres )
    # acquired_center = np.arange((yres - 2) / 2 - (calib / 2 - 1), yres / 2 + calib / 2, step=1,
    #                             dtype='int')
    # acquired_peri = np.random.randint(0, (yres - 1), num_peri, dtype='int')
    # mask = np.zeros((xres, yres), dtype=np.float32)
    # mask[:, acquired_center] = 1
    # mask[:, acquired_peri] = 1
    # mask = np.pad(mask, pad, 'constant', constant_values=0)
    masks.append(mask)

# Fermi window the data
[kx, ky] = np.meshgrid(np.linspace(-1, 1, act_xres), np.linspace(-1, 1, act_yres), indexing='ij')
# [kx, ky] = np.meshgrid(np.linspace(-1, 1, xres), np.linspace(-1, 1, yres), indexing='ij')
kr = (kx ** 2 + ky ** 2) ** 0.5
mask_truth = (1 / (1 + np.exp((np.abs(kr) - 0.9) / 0.1))).astype(np.float32)
mask_truth = np.pad(mask_truth, pad, 'constant', constant_values=0)
# mask_truth = torch.ones((xres, yres))
mask_truth = torch.tensor(mask_truth)
masks = torch.tensor(masks)

effective_acc = torch.count_nonzero(masks[0]) / torch.count_nonzero(mask_truth)
print(f'effective acc {effective_acc}')

contrasts = ['T1', 'T1post', 'T2', 'FLAIR']
# contrasts = ['T1']

for contrast in contrasts:

    globals()['mselistL' + contrast] = np.zeros(args.num_slices)
    globals()['ssimlistL' + contrast] = np.zeros(args.num_slices)
    globals()['scorelistL' + contrast] = np.zeros(args.num_slices)
    globals()['psnrlistL' + contrast] = np.zeros(args.num_slices)
    globals()['mselistMSE' + contrast] = np.zeros(args.num_slices)
    globals()['ssimlistMSE' + contrast] = np.zeros(args.num_slices)
    globals()['scorelistMSE' + contrast] = np.zeros(args.num_slices)
    globals()['psnrlistMSE' + contrast] = np.zeros(args.num_slices)
    globals()['mselistSSIM' + contrast] = np.zeros(args.num_slices)
    globals()['ssimlistSSIM' + contrast] = np.zeros(args.num_slices)
    globals()['scorelistSSIM' + contrast] = np.zeros(args.num_slices)
    globals()['psnrlistSSIM' + contrast] = np.zeros(args.num_slices)

    # try:
    #     os.remove(rf'I:\NYUbrain\poisson_acc1\ReconMSE_{contrast}.h5')
    #     os.remove(rf'I:\NYUbrain\poisson_acc1\ReconSSIM_{contrast}.h5')
    #     os.remove(rf'I:\NYUbrain\poisson_acc1\ReconL_{contrast}.h5')
    #     os.remove(rf'I:\NYUbrain\poisson_acc1\truth_{contrast}.h5')
    # except OSError:
    #     pass

    data_folder_val = os.path.join(args.data_dir, contrast)
    logging.basicConfig(filename=os.path.join(args.project_dir, f'metrics_compare_{contrast}.log'), filemode='w',
                        level=logging.INFO)

    validationset = DataGeneratorReconSlices(data_folder_val, rank_trained_on_mag=False,
                                             data_type='smap16', case_name=True)
    loader_V = DataLoader(dataset=validationset, batch_size=20, shuffle=True, pin_memory=True)

    with torch.no_grad():
        tt = time.time()
        for i, data in enumerate(loader_V, 0):
            smaps, kspace, fname = data
            smaps_sl = torch.clone(smaps[0]).to(device)
            kspace_sl = torch.clone(kspace[0]).to(device)
            mask_idx = np.random.randint(args.num_slices)
            mask_torch = masks[mask_idx].to(device)
            kspaceU_sl = kspace_sl * mask_torch

            # Get truth
            im_sl = sense_adjoint(smaps_sl, kspace_sl * mask_truth.to(device))

            # Get zerofilled image to estimate max
            imU_sl = sense_adjoint(smaps_sl, kspaceU_sl * mask_truth.to(device))

            # Scale based on max value
            scale = 1.0 / torch.max(torch.abs(imU_sl))
            im_sl *= scale
            kspaceU_sl *= scale
            kspace_sl *= scale

            imEst = torch.zeros_like(im_sl)
            imEstL = ReconModel(imEst, kspaceU_sl, smaps_sl, mask_torch)
            imEstMSE = ReconModelMSE(imEst, kspaceU_sl, smaps_sl, mask_torch)
            imEstSSIM = ReconModelSSIM(imEst, kspaceU_sl, smaps_sl, mask_torch)

            # crop to square
            width = im_sl.shape[2]
            height = im_sl.shape[1]
            idxL = int((height - width) / 2)
            idxR = int(idxL + width)
            im_sl = im_sl[:, idxL:idxR, :]
            imEstL = imEstL[:, idxL:idxR, :]
            imEstMSE = imEstMSE[:, idxL:idxR, :]
            imEstSSIM = imEstSSIM[:, idxL:idxR, :]

            # flipud
            im_sl = torch.flip(im_sl, dims=(0, 1))
            imEstL = torch.flip(imEstL, dims=(0, 1))
            imEstMSE = torch.flip(imEstMSE, dims=(0, 1))
            imEstSSIM = torch.flip(imEstSSIM, dims=(0, 1))

            # with h5py.File(os.path.join(r'I:\NYUbrain\poisson_acc1', f'ReconMSE_{contrast}.h5'), 'a') as hf:
            #     hf.create_dataset(f"{fname}_mag", data=np.abs(np.squeeze(imEstMSE.cpu().numpy())))
            #     hf.create_dataset(f"{fname}_phase", data=np.angle(np.squeeze(imEstMSE.cpu().numpy())))
            # with h5py.File(os.path.join(r'I:\NYUbrain\poisson_acc1', f'ReconSSIM_{contrast}.h5'), 'a') as hf:
            #     hf.create_dataset(f"{fname}_mag", data=np.abs(np.squeeze(imEstSSIM.cpu().numpy())))
            #     hf.create_dataset(f"{fname}_phase", data=np.angle(np.squeeze(imEstSSIM.cpu().numpy())))
            # with h5py.File(os.path.join(r'I:\NYUbrain\poisson_acc1', f'ReconL_{contrast}.h5'), 'a') as hf:
            #     hf.create_dataset(f"{fname}_mag", data=np.abs(np.squeeze(imEstL.cpu().numpy())))
            #     hf.create_dataset(f"{fname}_phase", data=np.angle(np.squeeze(imEstL.cpu().numpy())))
            # with h5py.File(os.path.join(r'I:\NYUbrain\poisson_acc1', f'truth_{contrast}.h5'), 'a') as hf:
            #     hf.create_dataset(f"{fname}_mag", data=np.abs(np.squeeze(im_sl).cpu().numpy()))
            #     hf.create_dataset(f"{fname}_phase", data=np.angle(np.squeeze(im_sl).cpu().numpy()))

            # MSE
            mseL = (mseloss_fcn(imEstL, im_sl)).squeeze().cpu().numpy()
            mseMSE = (mseloss_fcn(imEstMSE, im_sl)).squeeze().cpu().numpy()
            mseSSIM = (mseloss_fcn(imEstSSIM, im_sl)).squeeze().cpu().numpy()
            globals()['mselistL' + contrast][i] = mseL
            globals()['mselistMSE' + contrast][i] = mseMSE
            globals()['mselistSSIM' + contrast][i] = mseSSIM

            # PSNR
            max_truth = torch.abs(im_sl).max().cpu().numpy()
            psnrL = 10* np.log10(max_truth**2/(mseL/(width * height)))
            psnrMSE = 10* np.log10(max_truth**2/(mseMSE/(width * height)))
            psnrSSIM = 10* np.log10(max_truth**2/(mseSSIM/(width * height)))
            globals()['psnrlistL' + contrast][i] = psnrL
            globals()['psnrlistMSE' + contrast][i] = psnrMSE
            globals()['psnrlistSSIM' + contrast][i] = psnrSSIM

            # SSIM
            ssim_module = SSIM()
            ssimL = ssim_module(imEstL[(None,)], im_sl[(None,)]).squeeze().cpu().numpy()
            ssimMSE = ssim_module(imEstMSE[(None,)], im_sl[(None,)]).squeeze().cpu().numpy()
            ssimSSIM = ssim_module(imEstSSIM[(None,)], im_sl[(None,)]).squeeze().cpu().numpy()
            globals()['ssimlistL' + contrast][i] = ssimL
            globals()['ssimlistMSE' + contrast][i] = ssimMSE
            globals()['ssimlistSSIM' + contrast][i] = ssimSSIM


            # score
            get_score(i, imEstL, im_sl, scorenets,  globals()['scorelistL' + contrast], rank_trained_on_mag=False, augmentation=False)
            get_score(i, imEstMSE, im_sl, scorenets,  globals()['scorelistMSE' + contrast], rank_trained_on_mag=False, augmentation=False)
            get_score(i, imEstSSIM, im_sl, scorenets,  globals()['scorelistSSIM' + contrast], rank_trained_on_mag=False, augmentation=False)

            if i >= (args.num_slices-1):
                break


    globals()['mseMeanMSE' +contrast]= np.mean(globals()['mselistMSE' + contrast])
    globals()['ssimMeanMSE' +contrast]= np.mean(globals()['ssimlistMSE' + contrast])
    globals()['scoreMeanMSE' +contrast]= np.mean(globals()['scorelistMSE' + contrast])
    globals()['psnrMeanMSE' +contrast]= np.mean(globals()['psnrlistMSE' + contrast])
    globals()['mseStdMSE' +contrast]= np.std(globals()['mselistMSE' + contrast])
    globals()['ssimStdMSE' +contrast]= np.std(globals()['ssimlistMSE' + contrast])
    globals()['scoreStdMSE' +contrast]= np.std(globals()['scorelistMSE' + contrast])
    globals()['psnrStdMSE' +contrast]= np.std(globals()['psnrlistMSE' + contrast])

    globals()['mseMeanSSIM' +contrast]= np.mean(globals()['mselistSSIM' + contrast])
    globals()['ssimMeanSSIM' +contrast]= np.mean(globals()['ssimlistSSIM' + contrast])
    globals()['scoreMeanSSIM' +contrast]= np.mean(globals()['scorelistSSIM' + contrast])
    globals()['psnrMeanSSIM' +contrast]= np.mean(globals()['psnrlistSSIM' + contrast])
    globals()['mseStdSSIM' +contrast]= np.std(globals()['mselistSSIM' + contrast])
    globals()['ssimStdSSIM' +contrast]= np.std(globals()['ssimlistSSIM' + contrast])
    globals()['scoreStdSSIM' +contrast]= np.std(globals()['scorelistSSIM' + contrast])
    globals()['psnrStdSSIM' +contrast]= np.std(globals()['psnrlistSSIM' + contrast])

    globals()['mseMeanL' +contrast]= np.mean(globals()['mselistL' + contrast])
    globals()['ssimMeanL' +contrast]= np.mean(globals()['ssimlistL' + contrast])
    globals()['scoreMeanL' +contrast]= np.mean(globals()['scorelistL' + contrast])
    globals()['psnrMeanL' +contrast]= np.mean(globals()['psnrlistL' + contrast])
    globals()['mseStdL' +contrast]= np.std(globals()['mselistL' + contrast])
    globals()['ssimStdL' +contrast]= np.std(globals()['ssimlistL' + contrast])
    globals()['scoreStdL' +contrast]= np.std(globals()['scorelistL' + contrast])
    globals()['psnrStdL' +contrast]= np.std(globals()['psnrlistL' + contrast])

    globals()['df_mse'+contrast] = pd.DataFrame(
        {'Contrast': [f'{contrast}' for i in range(args.num_slices)], 'MSE': globals()['mselistMSE' + contrast], 'SSIM': globals()['mselistSSIM' + contrast],
         'Score': globals()['mselistL' + contrast]})
    globals()['df_mse'+contrast] = globals()['df_mse'+contrast][['Contrast', 'MSE', 'SSIM', 'Score']]

    globals()['df_ssim'+contrast] = pd.DataFrame(
        {'Contrast': [f'{contrast}' for i in range(args.num_slices)], 'MSE': globals()['ssimlistMSE' + contrast], 'SSIM': globals()['ssimlistSSIM' + contrast],
         'Score': globals()['ssimlistL' + contrast]})
    globals()['df_ssim'+contrast] = globals()['df_ssim'+contrast][['Contrast', 'MSE', 'SSIM', 'Score']]

    globals()['df_score'+contrast] = pd.DataFrame(
        {'Contrast': [f'{contrast}' for i in range(args.num_slices)], 'MSE': globals()['scorelistMSE' + contrast], 'SSIM': globals()['scorelistSSIM' + contrast],
         'Score': globals()['scorelistL' + contrast]})
    globals()['df_score'+contrast] = globals()['df_score'+contrast][['Contrast', 'MSE', 'SSIM', 'Score']]
    # logging.info(f'--- {contrast} ---')
    # logging.info('----- recon trained on mse -----')
    # logging.info(f'mse mean {np.mean(mselistMSE)}, std {np.std(mselistMSE)}')
    # logging.info(f'ssim mean {np.mean(ssimlistMSE)}, std {np.std(ssimlistMSE)}')
    # logging.info(f'score mean {np.mean(scorelistMSE)}, std {np.std(scorelistMSE)}')
    # logging.info(f'psnr mean {np.mean(psnrlistMSE)}, std {np.std(psnrlistMSE)}')
    #
    # logging.info('--- recon trained on ssim ---')
    # logging.info(f'mse mean {np.mean(mselistSSIM)}, std {np.std(mselistSSIM)}')
    # logging.info(f'ssim mean {np.mean(ssimlistSSIM)}, std {np.std(ssimlistSSIM)}')
    # logging.info(f'score mean {np.mean(scorelistSSIM)}, std {np.std(scorelistSSIM)}')
    # logging.info(f'psnr mean {np.mean(psnrlistSSIM)}, std {np.std(psnrlistSSIM)}')
    #
    # logging.info('--- recon trained on LMI ---')
    # logging.info(f'mse mean {}, std {np.std(mselistL)}')
    # logging.info(f'ssim mean {np.mean(ssimlistL)}, std {np.std(ssimlistL)}')
    # logging.info(f'score mean {np.mean(scorelistL)}, std {np.std(scorelistL)}')
    # logging.info(f'psnr mean {np.mean(psnrlistL)}, std {np.std(psnrlistL)}')



# logging.info(f'total time {time.time()-tt} sec')
#
# # boxplot
# dict = {'MSE': mselistMSE.transpose().squeeze(), 'SSIM': mselistSSIM.transpose().squeeze(), 'Learned': mselistL.transpose().squeeze()}
# fig, ax = plt.subplots(figsize=(3,2),tight_layout=True)
# # ax.set_title('MSE')
# ax.boxplot(dict.values(), showfliers=False)
# ax.set_xlabel('Training metrics')
# ax.set_ylabel('MSE')
# ax.set_xticklabels(dict.keys())
# fig.savefig(os.path.join(args.recon_dir, f'metrics_compare_mse_{args.contrast}.png'))
#
# dict = {'MSE': ssimlistMSE.transpose().squeeze(), 'SSIM': ssimlistSSIM.transpose().squeeze(), 'Learned': ssimlistL.transpose().squeeze()}
# fig, ax = plt.subplots(figsize=(3,2),tight_layout=True)
# # ax.set_title('SSIM')
# ax.boxplot(dict.values(), showfliers=False)
# ax.set_xlabel('Training metrics')
# ax.set_ylabel('SSIM')
# ax.set_xticklabels(dict.keys())
# fig.savefig(os.path.join(args.recon_dir, f'metrics_compare_ssim_{args.contrast}.png'))
#
# dict = {'MSE': scorelistMSE.transpose().squeeze(), 'SSIM': scorelistSSIM.transpose().squeeze(), 'Learned': scorelistL.transpose().squeeze()}
# fig, ax = plt.subplots(figsize=(3,2),tight_layout=True)
# # ax.set_title('Learned')
# ax.boxplot(dict.values(), showfliers=False)
# ax.set_xlabel('Training metrics')
# ax.set_ylabel('Score')
# ax.set_xticklabels(dict.keys())
# fig.savefig(os.path.join(args.recon_dir, f'metrics_compare_learned_{args.contrast}.png'))
#
# dict = {'MSE': psnrlistMSE.transpose().squeeze(), 'SSIM': psnrlistSSIM.transpose().squeeze(), 'Learned': psnrlistL.transpose().squeeze()}
# fig, ax = plt.subplots(figsize=(3,2),tight_layout=True)
# # ax.set_title('PSNR')
# ax.boxplot(dict.values(), showfliers=False)
# ax.set_xlabel('Training metrics')
# ax.set_ylabel('PSNR')
# ax.set_xticklabels(dict.keys())
# fig.savefig(os.path.join(args.recon_dir, f'metrics_compare_psnr_{args.contrast}.png'))


import seaborn as sns
dd_mse=pd.melt(pd.concat([df_mseT1, df_mseT1post, df_mseT2, df_mseFLAIR], ignore_index=True),id_vars=['Contrast'],value_vars=['MSE','SSIM', 'Score'],var_name='ReconLoss')
plt.figure()
sns.boxplot(x='Contrast',y='value',data=dd_mse,hue='ReconLoss', width=0.6, showfliers=False, palette='colorblind')
# plt.savefig('evalMSE.png')

dd_ssim=pd.melt(pd.concat([df_ssimT1, df_ssimT1post, df_ssimT2, df_ssimFLAIR], ignore_index=True),id_vars=['Contrast'],value_vars=['MSE','SSIM', 'Score'],var_name='ReconLoss')
plt.figure()
sns.boxplot(x='Contrast',y='value',data=dd_ssim,hue='ReconLoss', width=0.6, showfliers=False, palette='colorblind')
# plt.savefig('evalSSIM.png')

dd_score=pd.melt(pd.concat([df_scoreT1, df_scoreT1post, df_scoreT2, df_scoreFLAIR], ignore_index=True),id_vars=['Contrast'],value_vars=['MSE','SSIM', 'Score'],var_name='ReconLoss')
plt.figure()
sns.boxplot(x='Contrast',y='value',data=dd_score,hue='ReconLoss', width=0.6, showfliers=False, palette='colorblind')
# plt.savefig('evalScore.png')