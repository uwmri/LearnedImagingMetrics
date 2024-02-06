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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Ntrial = randrange(10000, 20000)

    # Argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_folder', type=str,
                        default=r'D:\NYUbrain\singleslices',
                        help='Data path')
    parser.add_argument('--cv_folder', type=str,
                        default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn\CV-5',
                        help='Name of learned metric file')
    parser.add_argument('--log_dir', type=str,
                        default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020',
                        help='Directory to log files')
    parser.add_argument('--pname', type=str, default=f'chenwei_recon_{Ntrial}')
    parser.add_argument('--WHICH_LOSS', type=str, default=f'learned')
    parser.add_argument('--resume_train', action='store_true', default=False)
    parser.add_argument('--save_all_slices', action='store_true', default=False)
    parser.add_argument('--save_train_images', action='store_true', default=False)
    parser.add_argument('--L2CNN', action='store_true', default=True)
    parser.add_argument('--EFF', action='store_true', default=False)
    parser.add_argument('--SUB', action='store_true', default=True)
    args = parser.parse_args()

    rank_channel = 1
    rank_trained_on_mag = False
    BO = False
    smap_type = 'smap16'
    # Only choose 20-coil data for now
    Ncoils = 20
    xres = 768
    yres = 396
    act_xres = 512
    act_yres = 256

    data_folder_train = os.path.join( args.data_folder, 'train')
    data_folder_val = os.path.join( args.data_folder, 'val')

    if args.L2CNN:
        if args.SUB:
            metric_files = [ os.path.join(args.cv_folder, os.listdir(args.cv_folder)[0]),
                             os.path.join(args.cv_folder, os.listdir(args.cv_folder)[1]),
                             os.path.join(args.cv_folder, os.listdir(args.cv_folder)[2]),
                             os.path.join(args.cv_folder, os.listdir(args.cv_folder)[3]),
                             os.path.join(args.cv_folder, os.listdir(args.cv_folder)[4])]
    elif args.EFF:
        if args.SUB:
            metric_files = [
                r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_efficientnet\RankClassifier2216.pt', ]
        else:
            metric_files = [
                r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_efficientnet\RankClassifier9909.pt', ]

    logging.basicConfig(filename=os.path.join(args.log_dir,f'Recon_{Ntrial}.log'), filemode='w', level=logging.INFO)


    if args.WHICH_LOSS == 'perceptual':
        loss_perceptual = PerceptualLoss_VGG16()
        loss_perceptual.to(device)
    elif args.WHICH_LOSS == 'patchGAN':
        patchGAN = NLayerDiscriminator(input_nc=2)
        patchGAN.to(device)
    elif args.WHICH_LOSS == 'ssim':
        ssim_module = SSIM()
    elif args.WHICH_LOSS == 'learned':
        scorenets = []
        for name in metric_files:
            if args.L2CNN:
                ranknet = L2cnn(channels_in=1, channel_base=8, group_depth=5, train_on_mag=rank_trained_on_mag, subtract_truth=args.SUB)
            elif args.EFF:
                from IQNet import EfficientNet2chan
                ranknet = EfficientNet2chan()

            classifier = Classifier(ranknet)

            state = torch.load(name)
            classifier.load_state_dict(state['state_dict'], strict=True)
            classifier.eval()
            score = classifier.rank
            score.to(device)

            scorenets.append(score)


    # Fermi window the data
    pady = int(.5 * (yres - act_yres))
    padx = int(.5 * (xres - act_xres))
    pad = ((padx, xres - padx - act_xres), (pady, yres - pady - act_yres))
    [kx, ky] = np.meshgrid(np.linspace(-1, 1, act_xres), np.linspace(-1, 1, act_yres), indexing='ij')
    kr = (kx ** 2 + ky ** 2) ** 0.5
    mask_truth = (1 / (1 + np.exp((np.abs(kr) - 0.9) / 0.1))).astype(np.float32)
    mask_truth = np.pad(mask_truth, pad, 'constant', constant_values=0)
    mask_truth = torch.tensor(mask_truth)

    # Data generator
    Ntrain = 64
    Nval = 200
    BATCH_SIZE = 8
    SaveCaseName = True
    print('Loading datasets')
    trainingset = DataGeneratorReconSlices(data_folder_train, rank_trained_on_mag=rank_trained_on_mag,
                                     data_type=smap_type, case_name=SaveCaseName)
    loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, drop_last=True, num_workers=2)


    validationset = DataGeneratorReconSlices(data_folder_val, rank_trained_on_mag=rank_trained_on_mag,
                                     data_type=smap_type, case_name=SaveCaseName)
    loader_V = DataLoader(dataset=validationset, batch_size=1, shuffle=False, pin_memory=True)

    Denoiser = UNet(in_channels=1, out_channels=1, f_maps=64, depth=2,
                                     layer_order=['convolution', 'relu'],
                                     complex_kernel=False, complex_input=True,
                                     residual=True, scaled_residual=True)
    Denoiser.cuda()
    LR = 1e-4
    optimizer = optim.Adam(Denoiser.parameters(), lr=LR)
    lmbda = lambda epoch: 0.99
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)


    writer_train = SummaryWriter(os.path.join(args.log_dir, f'runs/denoise/train_{Ntrial}'))
    writer_val = SummaryWriter(os.path.join(args.log_dir, f'runs/denoise/val_{Ntrial}'))

    Nepoch = 400
    lossT = np.zeros(Nepoch)
    lossV = np.zeros(Nepoch)

    out_name = os.path.join(args.log_dir, f'runs/recon/Images_denoise{Ntrial}_{args.WHICH_LOSS}_eval_case.h5')
    try:
        os.remove(out_name)
    except OSError:
        pass

    for epoch in range(Nepoch):

        # Setup counter to keep track of loss
        train_avg = RunningAverage()
        train_avg_mse = RunningAverage()
        eval_avg = RunningAverage()
        eval_avg_mse = RunningAverage()

        Denoiser.train()

        i=-1
        for data in loader_T:
            i = i + 1
            if SaveCaseName:
                smaps, kspace, name = data
                # logging.info(f'training case {name}')
            else:
                smaps, kspace = data
            smaps = smaps.cuda()
            kspace = kspace.cuda()
            if i==0:
                break
            optimizer.zero_grad()

            for sl in range(smaps.shape[0]):

                # Clone to torch
                smaps_sl = torch.clone(smaps[sl]).to(device)  # ndarray on cuda (20, 768, 396), complex64
                kspace_sl = torch.clone(kspace[sl]).to(device)  # ndarray (20, 768, 396), complex64

                # truth
                im_sl = sense_adjoint(smaps_sl, kspace_sl * mask_truth.to(device))

                # crop to square
                width = im_sl.shape[-1]
                height = im_sl.shape[-2]
                idxL = int((height - width) / 2)
                idxR = int(idxL + width)
                im_sl = im_sl[:, idxL:idxR, :]
                # flipud
                im_sl = torch.flip(im_sl, dims=(0, 1))

                scale = 1.0 / torch.max(torch.abs(im_sl))
                im_sl *= scale

                im_noisy = add_gaussian2D(im_sl, gaussian_level=7e4, kedge_len=30)      # im_sl torch.Size([1, 396, 396])


                imEst = Denoiser(im_noisy.unsqueeze(0))     # to torch.Size([1, 1, 396, 396])
                loss_mse = NMSE(imEst, im_sl.detach())      # imEst torch.Size([1, 1, 396, 396])

                if args.EFF:
                    # image_corrupted1 = imEst.detach().squeeze().cpu().numpy()
                    # image_sense = im.squeeze().cpu().numpy()
                    # for i in range(BATCH_SIZE):
                    #     scale = np.sum(np.conj(image_corrupted1[i]).T * image_sense[i]) / np.sum(
                    #         np.conj(image_corrupted1[i]).T * image_corrupted1[i])
                    #     imEst[i] = imEst[i] * scale
                    imEst = torch.view_as_real(imEst.squeeze()).unsqueeze(0).permute((0, -1, 1, 2))
                    im_sl = torch.view_as_real(im_sl.squeeze()).unsqueeze(0).permute((0, -1, 1, 2))
                    preprocess = transforms.Compose([transforms.Resize((224, 224))])
                    imEst = preprocess(imEst)
                    im_sl = preprocess(im_sl)

                loss = 0.0
                for score in scorenets:
                    loss += learnedloss_fcn(imEst, im_sl, score, rank_trained_on_mag=rank_trained_on_mag,
                                            augmentation=False, eff=args.EFF, sub=args.SUB)
                if args.L2CNN:
                    if args.SUB:
                        loss = loss / 5
                if args.EFF:
                    imEst = torch.view_as_complex(imEst.permute((0, 2, 3, 1))).unsqueeze(0)
                    im_sl = torch.view_as_complex(im_sl.permute((0, 2, 3, 1))).unsqueeze(0)

                loss.backward()

                train_avg_mse.update(loss_mse.detach().cpu().item(), n=1)
                train_avg.update(loss.detach().item(), n=1)
            optimizer.step()
            if i > Ntrain:
                break
        scheduler.step()

        with torch.no_grad():
            Denoiser.eval()
            im_nn_stack = []
            im_truth_stack = []
            for i, data in enumerate(loader_V, 0):

                if SaveCaseName:
                    smaps, kspace, name = data
                    # logging.info(f'val case {i} is {name}')
                else:
                    smaps, kspace = data
                smaps = smaps.cuda()
                kspace = kspace.cuda()
                if i==1:
                    break
                Nslices = smaps.shape[0]
                im = sense_adjoint(smaps, kspace * mask_truth.to(device))
                # crop to square
                width = im.shape[-1]
                height = im.shape[-2]
                idxL = int((height - width) / 2)
                idxR = int(idxL + width)
                im = im[:,:, idxL:idxR, :]
                # flipud
                im = torch.flip(im, dims=(0, 1,2))

                for sl in range(im.shape[0]):
                    scale = 1.0 / torch.max(torch.abs(im[sl]))
                    im[sl] *= scale
                im_noisy = add_gaussian2D(im[0], gaussian_level=7e4, kedge_len=30)
                imEst = Denoiser(im_noisy.unsqueeze(0))
                loss_mse = NMSE(imEst, im).detach()

                if args.EFF:
                    imEst = torch.view_as_real(imEst[0]).permute((0, -1, 1, 2))
                    im = torch.view_as_real(im[0]).permute((0, -1, 1, 2))
                    preprocess = transforms.Compose([transforms.Resize((224, 224))])
                    imEst = preprocess(imEst)
                    im = preprocess(im)

                loss = 0.0
                for score in scorenets:
                    loss += learnedloss_fcn(imEst, im, score, rank_trained_on_mag=rank_trained_on_mag,
                                            augmentation=False, eff=args.EFF, sub=args.SUB)
                if args.L2CNN:
                    if args.SUB:
                        loss = loss / 5

                if args.EFF:
                    imEst = torch.view_as_complex(imEst.permute((0, 2, 3, 1)))
                    im = torch.view_as_complex(im.permute((0, 2, 3, 1)))

                eval_avg.update(loss.detach().item(), n=1)
                eval_avg_mse.update(loss_mse.detach().cpu().item(), n=1)

                if i < 20:
                    if epoch==0:
                        im_truth_stack.append(np.squeeze(np.abs( im.detach().cpu().numpy())))
                    im_nn_stack.append(np.squeeze(np.abs(imEst.detach().cpu().numpy())))

                if i > Nval:
                    break

            with h5py.File(out_name, 'a') as hf:
                if epoch == 0:
                    hf.create_dataset(f"{epoch}_truth", data=np.stack(im_truth_stack))
                hf.create_dataset(f"{epoch}_recon", data=np.stack(im_nn_stack))

        writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
        writer_train.add_scalar('Loss', train_avg.avg(), epoch)
        writer_val.add_scalar('MSE Loss', eval_avg_mse.avg(), epoch)
        writer_train.add_scalar('MSE Loss', train_avg_mse.avg(), epoch)

        lossT[epoch] = train_avg.avg()
        lossV[epoch] = eval_avg.avg()
        state = {
            'state_dict': Denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss_train': lossT,
            'loss_cal': lossV
        }
        torch.save(state, os.path.join(args.log_dir, f'Denoise{Ntrial}_l2{args.L2CNN}_eff{args.EFF}.pt'))



# if __name__ == '__main__':
#     main()



