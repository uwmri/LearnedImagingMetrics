import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import sigpy as sp
import sigpy.mri as mri
import random

import torch
from torch.utils.data import DataLoader, Dataset

from utils.Recon_helper import *
from utils.model_helper import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_score(imEstL, im_sl, scorenets, scorelist, rank_trained_on_mag=False, augmentation=False):
    learnedlossL = 0.0
    for score in scorenets:
        learnedlossL += learnedloss_fcn(imEstL, im_sl, score, rank_trained_on_mag=rank_trained_on_mag, augmentation=augmentation)
    learnedlossL /= 5
    return scorelist.append(learnedlossL.cpu().numpy())

def get_mask(xres, yres, act_xres, act_yres, acc, calib, type='poisson'):
    if type == 'poisson':
        # mask = mri.poisson((xres, yres), accel=acc, calib=(calib, calib), crop_corner=True, return_density=False,
        #                    dtype='float32')

        mask = mri.poisson((act_xres, act_yres), accel=acc * 2, calib=(0, 0), crop_corner=True, return_density=False,
                           dtype='float32')
        pady = int(.5 * (yres - act_yres))
        padx = int(.5 * (xres - act_xres))
        print(mask.shape)
        print(f'padx = {(padx, xres - padx - mask.shape[0])}, {(pady, yres - pady - mask.shape[1])}')
        pad = ((padx, xres - padx - act_xres), (pady, yres - pady - act_yres))
        mask = np.pad(mask, pad, 'constant', constant_values=0)
    else:
        # random PE lines
        num_peri = int((xres * yres / acc - xres * calib) / xres)
        acquired_center = np.arange((yres - 2) / 2 - (calib / 2 - 1), yres / 2 + calib / 2, step=1,
                                    dtype='int')
        acquired_peri = np.random.randint(0, (yres - 1), num_peri, dtype='int')
        mask = np.zeros((xres, yres), dtype=np.float32)
        mask[:, acquired_center] = 1
        mask[:, acquired_peri] = 1
    return mask


class DataGeneratorReconSlices(Dataset):

    def __init__(self,data_folder, case_name=False):

        '''
        input: mask (768, 396) complex64
        output: all complex numpy array
                fully sampled kspace (rectangular), truth image (square), smaps (rectangular)
        '''


        self.scans = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
        random.shuffle(self.scans)

        print(f'Found {len(self.scans)} from {data_folder}')

        self.len = len(self.scans)
        self.data_folder = data_folder

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        fname = self.scans[idx]
        print(fname)
        with h5py.File(os.path.join(self.data_folder, fname),'r') as hf:
            print(hf.keys())
            kspace = np.array(hf['kspace'])
            kspace = sp.resize(kspace, kspace.shape[:-1]+(372,))  # pad to (sl, coil, 640, 372)
            kspace /= np.max(np.abs(kspace))/np.prod(kspace.shape[-2:])

            truth = np.array(hf['reconstruction_rss'])
        # Copy to torch
        kspace = torch.from_numpy(kspace)
        truth = torch.from_numpy(truth)

        return truth, kspace, fname

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_slices', type=int, default=200)
parser.add_argument('--acc', type=int, default=8)
parser.add_argument('--calib', type=int, default=0)
parser.add_argument('--mask_type', type=str, default='poisson')
parser.add_argument('--data_folder', type=str,
                    default=r'I:\NYUknee\multicoil_val')
parser.add_argument('--reconID_learned', type=str, default='1980')
parser.add_argument('--reconID_mse', type=str, default='8470')
parser.add_argument('--reconID_ssim', type=str, default='3987')
parser.add_argument('--recon_dir', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\recon_models')
parser.add_argument('--rank_dir', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn\CV-5')
parser.add_argument('--project_dir', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch')
args = parser.parse_args()

xres = 640
yres = 372
act_xres = 512
act_yres = 256
image_size = 320
central = args.calib
masks = []
for m in range(args.num_slices):
    masks.append(get_mask(xres, yres, act_xres, act_yres, args.acc, central, type=args.mask_type))
masks = torch.tensor(masks)

# Fermi window the data
# [kx, ky] = np.meshgrid(np.linspace(-1, 1, xres), np.linspace(-1, 1, yres), indexing='ij')
# kr = (kx ** 2 + ky ** 2) ** 0.5
# mask_truth = (1 / (1 + np.exp((np.abs(kr) - 0.9) / 0.1))).astype(np.float32)
pady = int(.5 * (yres - act_yres))
padx = int(.5 * (xres - act_xres))
print(f'padx = {(padx, xres - padx - act_xres)}, {(pady, yres - pady - act_yres)}')
pad = ((padx, xres - padx - act_xres), (pady, yres - pady - act_yres))
[kx, ky] = np.meshgrid(np.linspace(-1, 1, act_xres), np.linspace(-1, 1, act_yres), indexing='ij')
kr = (kx ** 2 + ky ** 2) ** 0.5
mask_truth = (1 / (1 + np.exp((np.abs(kr) - 0.9) / 0.1))).astype(np.float32)
mask_truth = np.pad(mask_truth, pad, 'constant', constant_values=0)
mask_truth = torch.tensor(mask_truth)

validationset = DataGeneratorReconSlices(args.data_folder, case_name=False)
loader = DataLoader(dataset=validationset, batch_size=1, shuffle=True,
                      pin_memory=True, drop_last=True)

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

metric_files = [f for f in os.listdir(args.rank_dir) if os.path.isfile(os.path.join(args.rank_dir, f))]
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

mselistL = []
ssimlistL = []
scorelistL = []
psnrlistL = []
mselistMSE = []
ssimlistMSE = []
scorelistMSE = []
psnrlistMSE = []
mselistSSIM = []
ssimlistSSIM = []
scorelistSSIM = []
psnrlistSSIM = []
logging.basicConfig(filename=os.path.join(args.project_dir, f'metrics_compare_knee.log'), filemode='w', level=logging.INFO)

try:
    os.remove(rf'I:\NYUknee\poisson_2x2\knee_ReconMSE.h5')
    os.remove(rf'I:\NYUknee\poisson_2x2\knee_ReconSSIM.h5')
    os.remove(rf'I:\NYUknee\poisson_2x2\knee_ReconL.h5')
    os.remove(rf'I:\NYUknee\poisson_2x2\knee_truth.h5')
except OSError:
    pass


with torch.no_grad():
    for i, data in enumerate(loader, 89):
        truth, kspace, fname = data

        slice_num = np.random.randint(kspace.shape[1])
        # get smaps
        smaps = get_smaps(sp.from_pytorch(kspace[0, slice_num, ...]), sp.Device(0), maxiter=30, method='jsense')
        smaps_sl = sp.to_pytorch(smaps).cuda()
        kspace_sl = torch.clone(kspace[0, slice_num, ...]).to(device)
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
        idxU = int((height - image_size) / 2)
        idxD = int(idxU + image_size)
        idxL = int((width - image_size) / 2)
        idxR = int(idxL + image_size)
        im_sl = im_sl[:, idxU:idxD, idxL:idxR]
        imEstL = imEstL[:, idxU:idxD, idxL:idxR]
        imEstMSE = imEstMSE[:, idxU:idxD, idxL:idxR]
        imEstSSIM = imEstSSIM[:, idxU:idxD, idxL:idxR]

        with h5py.File(os.path.join(r'I:\NYUknee\poisson_2x2', 'knee_ReconMSE.h5'), 'a') as hf:
            hf.create_dataset(f"{fname}_{slice_num}", data=np.abs(np.squeeze(imEstMSE.cpu().numpy())))
        with h5py.File(os.path.join(r'I:\NYUknee\poisson_2x2', 'knee_ReconSSIM.h5'), 'a') as hf:
            hf.create_dataset(f"{fname}_{slice_num}", data=np.abs(np.squeeze(imEstSSIM.cpu().numpy())))
        with h5py.File(os.path.join(r'I:\NYUknee\poisson_2x2', 'knee_ReconL.h5'), 'a') as hf:
            hf.create_dataset(f"{fname}_{slice_num}", data=np.abs(np.squeeze(imEstL.cpu().numpy())))
        with h5py.File(os.path.join(r'I:\NYUknee\poisson_2x2', 'knee_truth.h5'), 'a') as hf:
            hf.create_dataset(f"{fname}_{slice_num}", data=np.abs(np.squeeze(im_sl).cpu().numpy()))

        # MSE
        mseL = (mseloss_fcn(imEstL, im_sl)).squeeze().cpu().numpy()
        mseMSE = (mseloss_fcn(imEstMSE, im_sl)).squeeze().cpu().numpy()
        mseSSIM = (mseloss_fcn(imEstSSIM, im_sl)).squeeze().cpu().numpy()
        mselistL.append(mseL)
        mselistMSE.append(mseMSE)
        mselistSSIM.append(mseSSIM)

        # PSNR
        max_truth = torch.abs(im_sl).max().cpu().numpy()
        psnrL = 10 * np.log10(max_truth ** 2 / (mseL / (width * height)))
        psnrMSE = 10 * np.log10(max_truth ** 2 / (mseMSE / (width * height)))
        psnrSSIM = 10 * np.log10(max_truth ** 2 / (mseSSIM / (width * height)))
        psnrlistL.append(psnrL)
        psnrlistMSE.append(psnrMSE)
        psnrlistSSIM.append(psnrSSIM)

        # SSIM
        ssim_module = SSIM()
        ssimL = ssim_module(imEstL[(None,)], im_sl[(None,)]).squeeze().cpu().numpy()
        ssimMSE = ssim_module(imEstMSE[(None,)], im_sl[(None,)]).squeeze().cpu().numpy()
        ssimSSIM = ssim_module(imEstSSIM[(None,)], im_sl[(None,)]).squeeze().cpu().numpy()
        ssimlistL.append(ssimL)
        ssimlistMSE.append(ssimMSE)
        ssimlistSSIM.append(ssimSSIM)

        # score
        get_score(imEstL, im_sl, scorenets, scorelistL, rank_trained_on_mag=False, augmentation=False)
        get_score(imEstMSE, im_sl, scorenets, scorelistMSE, rank_trained_on_mag=False, augmentation=False)
        get_score(imEstSSIM, im_sl, scorenets, scorelistSSIM, rank_trained_on_mag=False, augmentation=False)

        if i >= (args.num_slices - 1):
            break


mselistL = np.array(mselistL)
ssimlistL = np.array(ssimlistL)
scorelistL = np.array(scorelistL)
psnrlistL = np.array(psnrlistL)
mselistMSE = np.array(mselistMSE)
ssimlistMSE = np.array(ssimlistMSE)
scorelistMSE = np.array(scorelistMSE)
psnrlistMSE = np.array(psnrlistMSE)
mselistSSIM = np.array(mselistSSIM)
ssimlistSSIM = np.array(ssimlistSSIM)
scorelistSSIM = np.array(scorelistSSIM)
psnrlistSSIM = np.array(psnrlistSSIM)

logging.info('--- recon trained on mse ---')
logging.info(f'mse mean {np.mean(mselistMSE)}, std {np.std(mselistMSE)}')
logging.info(f'ssim mean {np.mean(ssimlistMSE)}, std {np.std(ssimlistMSE)}')
logging.info(f'score mean {np.mean(scorelistMSE)}, std {np.std(scorelistMSE)}')
logging.info(f'psnr mean {np.mean(psnrlistMSE)}, std {np.std(psnrlistMSE)}')

logging.info('--- recon trained on ssim ---')
logging.info(f'mse mean {np.mean(mselistSSIM)}, std {np.std(mselistSSIM)}')
logging.info(f'ssim mean {np.mean(ssimlistSSIM)}, std {np.std(ssimlistSSIM)}')
logging.info(f'score mean {np.mean(scorelistSSIM)}, std {np.std(scorelistSSIM)}')
logging.info(f'psnr mean {np.mean(psnrlistSSIM)}, std {np.std(psnrlistSSIM)}')

logging.info('--- recon trained on LMI ---')
logging.info(f'mse mean {np.mean(mselistL)}, std {np.std(mselistL)}')
logging.info(f'ssim mean {np.mean(ssimlistL)}, std {np.std(ssimlistL)}')
logging.info(f'score mean {np.mean(scorelistL)}, std {np.std(scorelistL)}')
logging.info(f'psnr mean {np.mean(psnrlistL)}, std {np.std(psnrlistL)}')


# boxplot
dict = {'MSE': mselistMSE.transpose().squeeze(), 'SSIM': mselistSSIM.transpose().squeeze(), 'Learned': mselistL.transpose().squeeze()}
fig, ax = plt.subplots(figsize=(3,2),tight_layout=True)
# ax.set_title('MSE')
ax.boxplot(dict.values(), showfliers=False)
ax.set_xlabel('Training metrics')
ax.set_ylabel('MSE')
ax.set_xticklabels(dict.keys())
fig.savefig(os.path.join(args.recon_dir, f'metrics_compare_mse_knee.png'))

dict = {'MSE': ssimlistMSE.transpose().squeeze(), 'SSIM': ssimlistSSIM.transpose().squeeze(), 'Learned': ssimlistL.transpose().squeeze()}
fig, ax = plt.subplots(figsize=(3,2),tight_layout=True)
# ax.set_title('SSIM')
ax.boxplot(dict.values(), showfliers=False)
ax.set_xlabel('Training metrics')
ax.set_ylabel('SSIM')
ax.set_xticklabels(dict.keys())
fig.savefig(os.path.join(args.recon_dir, f'metrics_compare_ssim_knee.png'))

dict = {'MSE': scorelistMSE.transpose().squeeze(), 'SSIM': scorelistSSIM.transpose().squeeze(), 'Learned': scorelistL.transpose().squeeze()}
fig, ax = plt.subplots(figsize=(3,2),tight_layout=True)
# ax.set_title('Learned')
ax.boxplot(dict.values(), showfliers=False)
ax.set_xlabel('Training metrics')
ax.set_ylabel('Score')
ax.set_xticklabels(dict.keys())
fig.savefig(os.path.join(args.recon_dir, f'metrics_compare_learned_knee.png'))

dict = {'MSE': psnrlistMSE.transpose().squeeze(), 'SSIM': psnrlistSSIM.transpose().squeeze(), 'Learned': psnrlistL.transpose().squeeze()}
fig, ax = plt.subplots(figsize=(3,2),tight_layout=True)
# ax.set_title('PSNR')
ax.boxplot(dict.values(), showfliers=False)
ax.set_xlabel('Training metrics')
ax.set_ylabel('PSNR')
ax.set_xticklabels(dict.keys())
fig.savefig(os.path.join(args.recon_dir, f'metrics_compare_psnr_knee.png'))




