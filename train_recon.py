import numpy as np
import torch
#import torchvision
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch.linalg

import matplotlib
matplotlib.use('TKAgg')
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
from IQNet import *
from utils.Recon_helper import *
from utils.ISOResNet import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *
from random import randrange
#from fastmri.models.varnet import *



def main():

    # Network parameters
    INNER_ITER = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    spdevice = sp.Device(0)
    Ntrial = randrange(10000)

    # Argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_folder', type=str,
                        default=r'D:\NYUbrain\singleslices',
                        help='Data path')
    parser.add_argument('--metric_file', type=str,
                        default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn\CV-5\RankClassifier326.pt',
                        help='Name of learned metric file')
    parser.add_argument('--log_dir', type=str,
                        default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020',
                        help='Directory to log files')
    parser.add_argument('--pname', type=str, default=f'chenwei_recon_{Ntrial}')
    parser.add_argument('--resume_train', action='store_true', default=False)
    parser.add_argument('--save_all_slices', action='store_true', default=False)
    parser.add_argument('--save_train_images', action='store_true', default=False)
    args = parser.parse_args()

    log_dir = args.log_dir
    data_folder_train = os.path.join( args.data_folder, 'train')
    data_folder_val = os.path.join( args.data_folder, 'val')

    metric_file = args.metric_file
    resume_train = args.resume_train
    saveAllSl = args.save_all_slices
    saveTrainIm = args.save_train_images

    cv_folder = r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn\CV-5'
    metric_files = [ os.path.join(cv_folder, os.listdir(cv_folder)[0]),
                     os.path.join(cv_folder, os.listdir(cv_folder)[1]),
                     os.path.join(cv_folder, os.listdir(cv_folder)[2]),
                     os.path.join(cv_folder, os.listdir(cv_folder)[3]),
                     os.path.join(cv_folder, os.listdir(cv_folder)[4])]
    # metric_files = [ 'H:/LearnedImageMetric/ImagePairs_Pack_04032020/RankClassifier9644.pt', ]

    # load RankNet
    try:
        import setproctitle
        setproctitle.setproctitle(args.pname)
        print(f'Setting program name to {args.pname}')
    except:
        print('setproctitle not installled,unavailable, or failed')

    rank_channel = 1
    rank_trained_on_mag = False
    BO = False

    logging.basicConfig(filename=os.path.join(log_dir,f'Recon_{Ntrial}.log'), filemode='w', level=logging.INFO)

    WHICH_LOSS = 'learned'
    if WHICH_LOSS == 'perceptual':
        loss_perceptual = PerceptualLoss_VGG16()
        loss_perceptual.to(device)
    elif WHICH_LOSS == 'patchGAN':
        patchGAN = NLayerDiscriminator(input_nc=2)
        patchGAN.to(device)
    elif WHICH_LOSS == 'ssim':
        ssim_module = SSIM()

    if WHICH_LOSS == 'learned':
        scorenets = []
        for name in metric_files:
            # ranknet = L2cnn(channels_in=rank_channel, channel_base=16, train_on_mag=rank_trained_on_mag)
            # ranknet = L2cnn(channels_in=1, channel_base=8, group_depth=1, train_on_mag=rank_trained_on_mag)
            ranknet = L2cnn(channels_in=1, channel_base=8, group_depth=5, train_on_mag=rank_trained_on_mag)

            classifier = Classifier(ranknet)

            state = torch.load(name)
            classifier.load_state_dict(state['state_dict'], strict=True)
            classifier.eval()
            score = classifier.rank
            score.to(device)

            scorenets.append(score)

    smap_type = 'smap16'

    # Only choose 20-coil data for now
    Ncoils = 20
    xres = 768
    yres = 396
    act_xres = 512
    act_yres = 256

    acc = 3
    WHICH_MASK = 'poisson'
    NUM_MASK = 64
    logging.info(f'Acceleration = {acc}, {WHICH_MASK} mask, {NUM_MASK}')

    masks = []
    for m in range(NUM_MASK):
        if WHICH_MASK == 'poisson':
            mask = mri.poisson((act_xres, act_yres), accel=acc * 2, calib=(0, 0), crop_corner=True, return_density=False,
                               dtype='float32', seed=None)

        elif WHICH_MASK == 'randLines':
            length_center = 20
            num_peri = int((xres * yres / acc - xres * length_center) / xres)
            acquired_center = np.arange((yres - 2) / 2 - (length_center / 2 - 1), yres / 2 + length_center / 2, step=1,
                                        dtype='int')
            acquired_peri = np.random.randint(0, (yres - 1), num_peri, dtype='int')
            mask = np.zeros((xres, yres), dtype=np.float32)
            mask[:, acquired_center] = 1
            mask[:, acquired_peri] = 1
        else:
            mask = np.ones((act_xres, act_yres), dtype=np.float32)

        pady = int(.5 * (yres - mask.shape[1]))
        padx = int(.5 * (xres - mask.shape[0]))
        print(mask.shape)
        print(f'padx = {(padx, xres - padx - mask.shape[0])}, {(pady, yres - pady - mask.shape[1])}')
        pad = ((padx, xres - padx - mask.shape[0]), (pady, yres - pady - mask.shape[1]))
        mask = np.pad(mask, pad, 'constant', constant_values=0)

        masks.append(mask)


    # Fermi window the data
    [kx, ky] = np.meshgrid(np.linspace(-1, 1, act_xres), np.linspace(-1, 1, act_yres), indexing='ij')
    kr = (kx ** 2 + ky ** 2) ** 0.5
    mask_truth = (1 / (1 + np.exp((np.abs(kr) - 0.9) / 0.1))).astype(np.float32)
    mask_truth = np.pad(mask_truth, pad, 'constant', constant_values=0)

    mask_truth = torch.tensor(mask_truth)
    masks = torch.tensor(masks)

    # Data generator
    Ntrain = 64
    Nval = 200
    BATCH_SIZE = 32
    SaveCaseName = True
    print('Loading datasets')
    trainingset = DataGeneratorReconSlices(data_folder_train, rank_trained_on_mag=rank_trained_on_mag,
                                     data_type=smap_type, case_name=SaveCaseName)
    loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, drop_last=True, num_workers=2)


    validationset = DataGeneratorReconSlices(data_folder_val, rank_trained_on_mag=rank_trained_on_mag,
                                     data_type=smap_type, case_name=SaveCaseName)
    loader_V = DataLoader(dataset=validationset, batch_size=1, shuffle=False, pin_memory=True)

    print('Setting Model')

    if resume_train:
        #recon_file = '/raid/DGXUserDataRaid/cxt004/NYUbrain/Recon6680_learned.pt'
        #recon_file = r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\Recon8314_learned.pt'
        #recon_file = 'E:\LearnedImageMetric\saved_recon_model\Recon133_learned.pt'
        #recon_file = 'E:\LearnedImageMetric\runs\recon\Recon8959_learned.pt'
        recon_file = 'H:\LearnedImageMetric\runs\recon\Recon5046_learned.pt'

        recon_file = 'H:/LearnedImageMetric/runs/recon/Recon5046_learned.pt'
        ReconModel = MoDL(inner_iter=INNER_ITER, DENOISER='unet')

        state = torch.load(recon_file)
        ReconModel.load_state_dict(state['state_dict'], strict=True)
    else:
        denoiser = 'unet'
        logging.info(f'denoiser is {denoiser}')

        ReconModel = MoDL(inner_iter=INNER_ITER, DENOISER=denoiser)
        logging.info(f'MoDL, inner iter = {INNER_ITER}')

    ReconModel.to(device)

    LR = 1e-4
    optimizer = optim.Adam(ReconModel.parameters(), lr=LR)
    lmbda = lambda epoch: 0.99
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)


    writer_train = SummaryWriter(os.path.join(log_dir,f'runs/recon/train_{Ntrial}'))
    writer_val = SummaryWriter(os.path.join(log_dir,f'runs/recon/val_{Ntrial}'))




    Nepoch = 1001
    epochMSE = 0
    logging.info(f'MSE for first {epochMSE} epochs then switch to learned')
    lossT = np.zeros(Nepoch)
    lossV = np.zeros(Nepoch)

    out_name_train = os.path.join(log_dir,f'Images_training{Ntrial}_{WHICH_LOSS}_train.h5')
    try:
        os.remove(out_name_train)
    except OSError:
        pass

    out_name = os.path.join(log_dir, f'Images_training{Ntrial}_{WHICH_LOSS}_eval_case.h5')
    try:
        os.remove(out_name)
    except OSError:
        pass


    scaler = torch.cuda.amp.GradScaler()

    logging.info(f'Adam, lr = {LR}')
    logging.info('case averaged loss')
    logging.info('Denoiser = cnn_shortcut')
    logging.info(f'{Ntrain} cases training, {Nval} cases validation')
    logging.info(f'loss = {WHICH_LOSS}')
    logging.info(f'acc = {acc}, mask is {WHICH_MASK}')

    for epoch in range(Nepoch):

        # Setup counter to keep track of loss
        train_avg = RunningAverage()
        train_avg_mse = RunningAverage()
        eval_avg = RunningAverage()
        eval_avg_mse = RunningAverage()

        train_avg_mse0 = RunningAverage()
        eval_avg_mse0 = RunningAverage()

        loss_mseT = []
        loss_mseT0 = []
        loss_learnedT = []
        loss_mseV = []
        loss_mseV0 = []
        loss_learnedV = []

        ReconModel.train()
        tt = time.time()
        with torch.autograd.set_detect_anomaly(False):
            i = -1
            for data in loader_T:
                i = i + 1
                # print(f'-------------------------------beginning of training, epoch {epoch}-------------------------------')
                # print_mem()
                tstart_batch = time.time()
                if SaveCaseName:
                    smaps, kspace, name = data
                    logging.info(f'training case {name}')
                else:
                    smaps, kspace = data

                t_case = time.time()
                optimizer.zero_grad()

                loss_avg = 0.0

                # Loop over slices to reduce memory
                for sl in range(smaps.shape[0]):

                    t_sl = time.time()

                    # Clone to torch
                    smaps_sl = torch.clone(smaps[sl]).to(device)  # ndarray on cuda (20, 768, 396), complex64
                    kspace_sl = torch.clone(kspace[sl]).to(device)  # ndarray (20, 768, 396), complex64

                    # Get mask
                    mask_idx = np.random.randint(NUM_MASK)
                    mask_torch = masks[mask_idx].to(device)

                    # Move to torch
                    kspaceU_sl = kspace_sl * mask_torch

                    # Get truth on the fly
                    im_sl = sense_adjoint(smaps_sl, kspace_sl * mask_truth.to(device))

                    # Get zerofilled image to estimate max
                    imU_sl = sense_adjoint(smaps_sl, kspaceU_sl * mask_truth.to(device))

                    # denoiser
                    if INNER_ITER==0:
                        scale_im = torch.sum(torch.conj(imU_sl).permute((1,0,2)) * im_sl) / torch.sum(
                            torch.conj(imU_sl).permute((1,0,2))  * imU_sl)
                        imU_sl = scale_im * imU_sl

                    ########################################## MoDL recon #############################################
                    # Scale based on max value
                    scale = 1.0/torch.max(torch.abs(imU_sl))
                    im_sl *= scale
                    kspaceU_sl *= scale
                    kspace_sl *= scale

                    with torch.cuda.amp.autocast():

                        # Get PyTorch functions
                        t = time.time()
                        if INNER_ITER == 0:
                            imEst = imU_sl.clone()
                        else:
                            imEst = torch.zeros_like(im_sl)
                        imEst2 = ReconModel(imEst, kspaceU_sl, smaps_sl, mask_torch)  # (768, 396, 2)
                        t = time.time()
                    ########################################## MoDL recon #############################################

                        # crop to square
                        width = im_sl.shape[2]
                        height = im_sl.shape[1]
                        if width < 320:
                            im_sl = im_sl[:, 160:480, :]
                            imEst2 = imEst2[:, 160:480, :]
                        else:
                            idxL = int((height - width) / 2)
                            idxR = int(idxL + width)
                            im_sl = im_sl[:, idxL:idxR, :]
                            imEst2 = imEst2[:, idxL:idxR, :]

                        # flipud
                        im_sl = torch.flip(im_sl, dims=(0, 1))
                        imEst2 = torch.flip(imEst2, dims=(0, 1))

                        loss_mse_tensor = mseloss_fcn(imEst2, im_sl).detach()
                        loss_mse_tensor0 = mseloss_fcn0(imEst2, im_sl).detach()

                        if WHICH_LOSS == 'mse':
                            loss = mseloss_fcn(imEst2, im_sl)
                        elif WHICH_LOSS == 'ssim':
                            loss = torch.mean(1 - ssim_module(imEst2.unsqueeze(dim=0), im_sl.unsqueeze(dim=0)))
                        elif WHICH_LOSS == 'perceptual':
                            loss = loss_perceptual(imEst2, im_sl)
                        elif WHICH_LOSS == 'patchGAN':
                            loss = loss_GAN(imEst2, im_sl, patchGAN)
                        else:
                            if epoch < epochMSE:
                                loss = mseloss_fcn(imEst2, im_sl) * 5e2
                            else:
                                loss = 0.0
                                for score in scorenets:
                                    loss += learnedloss_fcn(imEst2, im_sl, score, rank_trained_on_mag=rank_trained_on_mag, augmentation=False)

                    scaler.scale(loss).backward()

                    t = time.time()

                    if saveAllSl:
                        imEstpltT = imEst2.detach().cpu()
                        truthpltT = im_sl.detach().cpu()
                        with h5py.File(f'ReconTraining_{Ntrial}.h5', 'a') as hf:
                            hf.create_dataset(f"Tcase_{i}_{sl}", data=np.squeeze(imEstpltT.numpy()))
                            hf.create_dataset(f"truth_Tcase_{i}_{sl}", data=np.squeeze(truthpltT.numpy()))

                    if saveTrainIm and i == 0 and sl == 4:
                        truthplt = im_sl.detach().cpu()
                        perturbed = sense_adjoint(smaps_sl, kspaceU_sl)
                        noisyplt = perturbed.detach().cpu()
                        noisyplt = noisyplt[:, idxL:idxR, :]
                        del perturbed
                        imEstplt = imEst2.detach().cpu()

                        if epoch == 0:

                            mask_gpu = sp.from_pytorch(mask_torch)

                            kspaceU_sl_gpu = sp.to_device(sp.from_pytorch(kspaceU_sl.cpu()), sp.Device(0))
                            smaps_sl_gpu = sp.to_device(sp.from_pytorch(smaps_sl.cpu()), sp.Device(0))
                            imSense = sp.mri.app.SenseRecon(kspaceU_sl_gpu, smaps_sl_gpu, weights=mask_gpu, lamda=.01,
                                                            max_iter=20, device=spdevice).run()
                            imSense = imSense[idxL:idxR, :]
                            # L1-wavelet
                            imL1 = sp.mri.app.L1WaveletRecon(kspaceU_sl_gpu, smaps_sl_gpu, weights=mask_gpu, lamda=.001,
                                                             max_iter=20, device=spdevice).run()
                            imL1 = imL1[idxL:idxR, :]

                        with h5py.File(out_name_train, 'a') as hf:
                            if epoch == 0:
                                hf.create_dataset(f"{epoch}_truth", data=np.squeeze(np.abs(truthplt.numpy())))
                                hf.create_dataset(f"{epoch}_truth_re", data=np.squeeze(np.real(truthplt.numpy())))
                                hf.create_dataset(f"{epoch}_truth_im", data=np.squeeze(np.imag(truthplt.numpy())))
                                hf.create_dataset(f"{epoch}_FT", data=np.squeeze(np.abs(noisyplt.numpy())))
                                hf.create_dataset(f"{epoch}_Sense", data=np.squeeze(np.abs(imSense.get())))
                                hf.create_dataset(f"{epoch}_L1", data=np.squeeze(np.abs(imL1.get())))
                                hf.create_dataset(f"p{epoch}_truth", data=np.squeeze(np.angle(truthplt.numpy())))
                                hf.create_dataset(f"p{epoch}_FT", data=np.squeeze(np.angle(noisyplt.numpy())))
                                hf.create_dataset(f"p{epoch}_Sense", data=np.squeeze(np.angle(imSense.get())))
                                hf.create_dataset(f"p{epoch}_L1", data=np.squeeze(np.angle(imL1.get())))

                            hf.create_dataset(f"p{epoch}_recon", data=np.squeeze(np.angle(imEstplt.numpy())))
                            hf.create_dataset(f"{epoch}_recon", data=np.squeeze(np.abs(imEstplt.numpy())))
                            if epoch % 20 == 0:
                                hf.create_dataset(f"{epoch}_recon_re", data=np.squeeze(np.real(imEstplt.numpy())))
                                hf.create_dataset(f"{epoch}_recon_im", data=np.squeeze(np.imag(imEstplt.numpy())))
                        del truthplt, noisyplt, imEstplt

                    if WHICH_LOSS == 'learned':
                        with torch.no_grad():
                            loss_learnedT.append(loss.detach().item())
                            loss_mseT.append(loss_mse_tensor.detach().cpu().item())
                            loss_mseT0.append(loss_mse_tensor0.detach().cpu().item())

                    loss_avg += loss.detach().item()
                    train_avg.update(loss.detach().item())
                    train_avg_mse.update(loss_mse_tensor.detach().cpu().item())
                    train_avg_mse0.update(loss_mse_tensor0.detach().cpu().item())

                # Slice loop
                #optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                print(f'Step {i} of {len(loader_T)}, took {time.time() - t_case}s, loss {loss_avg}, avg = {train_avg.avg()}')

                if i > Ntrain:
                    break

        scheduler.step()
        print(f'Epoch took {time.time() - tstart_batch}')

        with torch.no_grad():
            ReconModel.eval()
            save_count = 0

            im_nn_stack = []
            im_truth_stack = []
            im_L1_stack = []
            im_sense_stack = []
            im_ft_stack = []

            for i, data in enumerate(loader_V, 0):
                if SaveCaseName:
                    smaps, kspace, name = data
                    logging.info(f'val case {i} is {name}')
                else:
                    smaps, kspace = data

                Nslices = smaps.shape[0]

                for sl in range(smaps.shape[0]):
                    smaps_sl = torch.clone(smaps[sl]).to(device)
                    kspace_sl = torch.clone(kspace[sl]).to(device)

                    # Get mask
                    mask_torch = masks[0].to(device)

                    kspaceU_sl = kspace_sl * mask_torch

                    # Get truth
                    im_sl = sense_adjoint(smaps_sl, kspace_sl * mask_truth.to(device))

                    # Get zerofilled image to estimate max
                    imU_sl = sense_adjoint(smaps_sl, kspaceU_sl * mask_truth.to(device))

                    # denoiser
                    if INNER_ITER==0:
                        scale_im = torch.sum(torch.conj(imU_sl).permute((1,0,2)) * im_sl) / torch.sum(
                            torch.conj(imU_sl).permute((1,0,2))  * imU_sl)
                        imU_sl = scale_im * imU_sl

                    # Scale based on max value
                    scale = 1.0/torch.max(torch.abs(imU_sl))
                    im_sl *= scale
                    kspaceU_sl *= scale
                    kspace_sl *= scale
                    if INNER_ITER == 0:
                        imEst = imU_sl.clone()
                    else:
                        imEst = torch.zeros_like(im_sl)

                    t = time.time()
                    imEst2 = ReconModel(imEst, kspaceU_sl, smaps_sl, mask_torch)

                    # crop to square
                    width = im_sl.shape[2]
                    height = im_sl.shape[1]
                    if width < 320:
                        im_sl = im_sl[:,160:480, :]
                        imEst2 = imEst2[:,160:480, :]
                    else:
                        idxL = int((height - width) / 2)
                        idxR = int(idxL + width)
                        im_sl = im_sl[:,idxL:idxR, :]
                        imEst2 = imEst2[:,idxL:idxR, :]

                    # flipud
                    im_sl = torch.flip(im_sl, dims=(0,1))
                    imEst2 = torch.flip(imEst2, dims=(0,1))

                    loss_mse_tensor = mseloss_fcn(imEst2, im_sl).detach()
                    loss_mse_tensor0 = mseloss_fcn0(imEst2, im_sl).detach()
                    if WHICH_LOSS == 'mse':
                        loss = mseloss_fcn(imEst2, im_sl)
                    elif WHICH_LOSS == 'ssim':
                        loss = torch.mean(1 - ssim_module(imEst2.unsqueeze(dim=0), im_sl.unsqueeze(dim=0)))
                    elif WHICH_LOSS == 'perceptual':
                        loss = loss_perceptual(imEst2, im_sl)
                    elif WHICH_LOSS == 'patchGAN':
                        loss = loss_GAN(imEst2, im_sl, patchGAN)
                    else:
                        if epoch < epochMSE:
                            loss = mseloss_fcn(imEst2, im_sl) *5e2
                        else:
                            loss = 0.0
                            for score in scorenets:
                                loss += learnedloss_fcn(imEst2, im_sl, score, rank_trained_on_mag=rank_trained_on_mag,
                                                        augmentation=False)

                    eval_avg.update(loss.detach().item(), n=BATCH_SIZE)
                    eval_avg_mse.update(loss_mse_tensor.detach().cpu().item())
                    eval_avg_mse0.update(loss_mse_tensor.detach().cpu().item())

                    if WHICH_LOSS == 'learned':
                        loss_learnedV.append(loss.detach().item())
                        loss_mseV.append(loss_mse_tensor.detach().item())
                        loss_mseV0.append(loss_mse_tensor0.detach().item())

                    if saveAllSl:
                        imEstpltV = imEst2.detach().cpu()
                        truthpltV = im_sl.detach().cpu()
                        with h5py.File(f'ReconTraining_{Ntrial}.h5', 'a') as hf:
                            hf.create_dataset(f"Vcase_{i}_{sl}", data=np.squeeze(imEstpltV.numpy()))
                            hf.create_dataset(f"truthVcase_{i}_{sl}", data=np.squeeze(truthpltV.numpy()))

                    if sl == 0 and i < 20:
                        truthplt = im_sl.detach().cpu()
                        perturbed = sense_adjoint(smaps_sl, kspaceU_sl)
                        noisyplt = perturbed.detach().cpu()
                        noisyplt = noisyplt[:, idxL:idxR,:]
                        del perturbed
                        imEstplt = imEst2.detach().cpu()
                        imEstfig = plt_recon(torch.squeeze(torch.abs(imEstplt)))
                        writer_val.add_figure('Recon_val', imEstfig, epoch)

                        # SENSE
                        if epoch == 0:
                            mask_gpu = sp.from_pytorch(mask_torch)

                            kspaceU_sl_gpu = sp.to_device(sp.from_pytorch(kspaceU_sl.cpu()), sp.Device(0))
                            smaps_sl_gpu = sp.to_device(sp.from_pytorch(smaps_sl.cpu()), sp.Device(0))
                            imSense = sp.mri.app.SenseRecon(kspaceU_sl_gpu, smaps_sl_gpu, weights=mask_gpu, lamda=.01,
                                                            max_iter=20, device=spdevice).run()
                            imSense = imSense[idxL:idxR,:]
                            # L1-wavelet
                            imL1 = sp.mri.app.L1WaveletRecon(kspaceU_sl_gpu, smaps_sl_gpu, weights=mask_gpu, lamda=.001,
                                                             max_iter=20, device=spdevice).run()
                            imL1 = imL1[idxL:idxR,:]

                            im_truth_stack.append(np.squeeze(np.abs(truthplt.numpy())))
                            im_L1_stack.append(np.squeeze(np.abs(imL1.get())))
                            im_sense_stack.append(np.squeeze(np.abs(imSense.get())))
                            im_ft_stack.append(np.squeeze(np.abs(noisyplt.numpy())))
                        im_nn_stack.append(np.squeeze(np.abs(imEstplt.numpy())))

                    del smaps_sl, kspaceU_sl, im_sl

                print(f'Val Step {i} of {len(loader_V)}, avg = {eval_avg_mse.avg()}')

                if i > Nval:
                    break

            # Export the stack examples
            with h5py.File(out_name, 'a') as hf:
                if epoch == 0:
                    im_truth_stack = np.stack(im_truth_stack)
                    im_L1_stack = np.stack(im_L1_stack)
                    im_sense_stack = np.stack(im_sense_stack)
                    im_ft_stack = np.stack(im_ft_stack)

                    hf.create_dataset(f"{epoch}_truth", data=im_truth_stack)
                    hf.create_dataset(f"{epoch}_FT", data=im_ft_stack)
                    hf.create_dataset(f"{epoch}_Sense", data=im_sense_stack)
                    hf.create_dataset(f"{epoch}_L1", data=im_L1_stack)

                im_nn_stack = np.stack(im_nn_stack)
                hf.create_dataset(f"{epoch}_recon", data=im_nn_stack)

        logging.info(
            f'epoch {epoch} took {(time.time() - tt) / 60:.2f} min, '
            f'train = {train_avg.avg()}, eval = {eval_avg.avg()}, '
            f'mse eval = {eval_avg_mse.avg()}')

        # torch.cuda.empty_cache()

        writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
        writer_train.add_scalar('Loss', train_avg.avg(), epoch)
        writer_val.add_scalar('MSE Loss', eval_avg_mse.avg(), epoch)
        writer_train.add_scalar('MSE Loss', train_avg_mse.avg(), epoch)
        writer_val.add_scalar('MSE Loss0', eval_avg_mse0.avg(), epoch)
        writer_train.add_scalar('MSE Loss0', train_avg_mse0.avg(), epoch)

        # logging.info('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
        lossT[epoch] = train_avg.avg()
        lossV[epoch] = eval_avg.avg()

        # Score diff vs MSE between images and truths
        if WHICH_LOSS == 'learned':
            loss_learnedT = np.array(loss_learnedT)
            loss_mseT = np.array(loss_mseT)
            lossplotT = plt_scoreVsMse(loss_learnedT, loss_mseT)
            writer_train.add_figure('Loss_learned_vs_mse', lossplotT, epoch)

            loss_learnedV = np.array(loss_learnedV)
            loss_mseV = np.array(loss_mseV)
            lossplotV = plt_scoreVsMse(loss_learnedV, loss_mseV)
            writer_val.add_figure('Loss_learned_vs_mse', lossplotV, epoch)

            loss_mseT0 = np.array(loss_mseT0)
            lossplotT0 = plt_scoreVsMse(loss_learnedT, loss_mseT0)
            writer_train.add_figure('Loss_learned_vs_mse0', lossplotT0, epoch)

            loss_mseV0 = np.array(loss_mseV0)
            lossplotV0= plt_scoreVsMse(loss_learnedV, loss_mseV0)
            writer_val.add_figure('Loss_learned_vs_mse0', lossplotV0, epoch)

        # save models
        state = {
            'state_dict': ReconModel.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss_train': lossT,
            'loss_cal': lossV
        }
        torch.save(state, os.path.join(log_dir, f'Recon{Ntrial}_{WHICH_LOSS}.pt'))

# if __name__ == '__main__':
#     main()