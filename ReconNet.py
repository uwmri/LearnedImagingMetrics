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
from utils.model_helper import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *
from random import randrange

mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()
xp = spdevice.xp
spdevice = sp.Device(0)

Ntrial = randrange(10000)

# load RankNet
DGX = False
if DGX:
    filepath_rankModel = Path('/raid/DGXUserDataRaid/cxt004/NYUbrain')
    filepath_train = Path('/raid/DGXUserDataRaid/cxt004/NYUbrain')
    filepath_val = Path('/raid/DGXUserDataRaid/cxt004/NYUbrain')

    try:
        import setproctitle
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--pname', type=str, default=f'chenwei_recon_{Ntrial}')
        args = parser.parse_args()

        setproctitle.setproctitle(args.pname)
        print(f'Setting program name to {args.pname}')
    except:
        print('setproctitle not installled,unavailable, or failed')


else:
    filepath_rankModel = Path('I:/code/LearnedImagingMetrics_pytorch/Rank_NYU/ImagePairs_Pack_04032020')
    filepath_train = Path("I:/NYUbrain")
    filepath_val = Path("I:/NYUbrain")

    # On Kevins machine
    filepath_rankModel = Path('E:/LearnedImageMetric/ImagePairs_Pack_04032020')
    filepath_train = Path("Q:/LearnedImageMetric")
    filepath_val = Path("Q:/LearnedImageMetric")

    # Chenweis machine
    filepath_rankModel = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
    filepath_train = Path("I:/NYUbrain")
    filepath_val = Path("I:/NYUbrain")


rank_channel =1
rank_trained_on_mag = False
BO = False
file_rankModel = os.path.join(filepath_rankModel, "RankClassifier4217_pretrained.pt")
os.chdir(filepath_rankModel)

log_dir = filepath_rankModel
logging.basicConfig(filename=os.path.join(log_dir,f'Recon_{Ntrial}_dgx{DGX}.log'), filemode='w', level=logging.INFO)

ranknet = L2cnn(channels_in=rank_channel)
classifier = Classifier(ranknet)

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False
score = classifier.rank
score.cuda()

file_train = 'ksp_truths_smaps_train_lzf.h5'
file_val = 'ksp_truths_smaps_val_lzf.h5'
smap_type = 'smap16'

# Only choose 20-coil data for now
Ncoils = 20
xres = 768
yres = 396
act_xres = 512
act_yres = 256

acc = 8
WHICH_MASK = 'poisson'
logging.info(f'Acceleration = {acc}, {WHICH_MASK} mask')
if WHICH_MASK == 'poisson':
    # Sample elipsoid to not be overly optimistic
    mask = mri.poisson((act_xres, act_yres), accel=acc * 2, calib=(0, 0), crop_corner=True, return_density=False,
                       dtype='float32')
    pady = int(.5 * (yres - mask.shape[1]))
    padx = int(.5 * (xres - mask.shape[0]))
    print(mask.shape)
    print(f'padx = {(padx, xres - padx - mask.shape[0])}, {(pady, yres - pady - mask.shape[1])}')
    pad = ((padx, xres - padx - mask.shape[0]), (pady, yres - pady - mask.shape[1]))
    mask = np.pad(mask, pad, 'constant', constant_values=0)
    print(mask.shape)

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
    mask = np.ones((xres, yres), dtype=np.float32)

[kx, ky] = np.meshgrid(np.linspace(-1, 1, act_xres), np.linspace(-1, 1, act_yres), indexing='ij')
kr = (kx ** 2 + ky ** 2) ** 0.5
mask_truth = (1 / (1 + np.exp((np.abs(kr) - 0.9) / 0.1))).astype(np.float32)
mask_truth = np.pad(mask_truth, pad, 'constant', constant_values=0)
# print(mask_truth.shape)

plt.figure()
plt.imshow(mask_truth, cmap='gray')
plt.show()

plt.figure()
plt.imshow(mask)
plt.show()

mask_truth = sp.to_device(mask_truth, spdevice)  # Square here to account for sqrt in SENSE operator
mask_gpu = sp.to_device(mask, spdevice)

# Data generator
BATCH_SIZE = 1
prefetch_data = False
logging.info(f'Load train data from {filepath_train}')
trainingset = DataGeneratorRecon(filepath_train, file_train, rank_trained_on_mag=rank_trained_on_mag,
                                 data_type=smap_type)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=False)

logging.info(f'Load eval data from {filepath_val}')
validationset = DataGeneratorRecon(filepath_val, file_val, rank_trained_on_mag=rank_trained_on_mag,
                                   data_type=smap_type)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=False)


INNER_ITER = 5
ReconModel = MoDL(inner_iter=INNER_ITER)
ReconModel.cuda()
# torchsummary.summary(ReconModel.denoiser, input_size=(2,768,396), batch_size=16)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer_train = SummaryWriter(f'runs/recon/train_{Ntrial}')
writer_val = SummaryWriter(f'runs/recon/val_{Ntrial}')


WHICH_LOSS = 'learned'
if WHICH_LOSS == 'perceptual':
    loss_perceptual = PerceptualLoss_VGG16()
    loss_perceptual.cuda()
elif WHICH_LOSS == 'patchGAN':
    patchGAN = NLayerDiscriminator(input_nc=2)
    patchGAN.cuda()

Nepoch = 100
epochMSE = 0
logging.info(f'MSE for first {epochMSE} epochs then switch to learned')
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)

Ntrain = 27
Nval = 3

out_name = os.path.join(log_dir,f'Images_training{Ntrial}_{WHICH_LOSS}.h5')

LR = 1e-4
# optimizer = optim.SGD(ReconModel.parameters(), lr=LR, momentum=0.9)
optimizer = optim.Adam(ReconModel.parameters(), lr=LR)
if epochMSE != 0:
    lambda1 = lambda epoch: 1e1 if epoch<epochMSE else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

logging.info(f'Adam, lr = {LR}')
logging.info('case averaged loss')
logging.info('Denoiser = Unet')
logging.info(f'inner_iter = {INNER_ITER}')
logging.info(f'{Ntrain} cases training, {Nval} cases validation')
logging.info(f'loss = {WHICH_LOSS}')
logging.info(f'acc = {acc}, mask is {WHICH_MASK}')

# Get the scale for Denoiser
with torch.no_grad():
    da = 0.0
    for avg in range(10):
        x = torch.randn((1, 2, yres, xres), device='cuda')
        for iter in range(30):
            ein = torch.linalg.norm(x)
            y = ReconModel.call_denoiser(x)
            eout = torch.linalg.norm(y)
            x = y / eout
            # print(f'Scale {eout/ein}')
        # print(eout)
        da += eout / 10
    logging.info(f'Avg Denoiser scale = {da}')
    ReconModel.set_denoiser_scale( 0.9 / da)

for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()
    loss_mseT = []
    loss_learnedT = []
    loss_mseV = []
    loss_learnedV = []

    ReconModel.train()
    tt = time.time()

    for i, data in enumerate(loader_T, 0):
        # print(f'-------------------------------beginning of training, epoch {epoch}-------------------------------')
        # print_mem()
        tstart_batch = time.time()

        smaps, im, kspace = data
        smaps = sp.to_device(smaps, device=spdevice)
        smaps = spdevice.xp.squeeze(smaps)

        kspace *= 1e6
        kspace = sp.to_device(kspace, device=spdevice)
        kspace = chan2_complex(spdevice.xp.squeeze(kspace))  # cupy array on cuda

        im = torch.squeeze(im, dim=0)  # torch.Size([16, 768, 396, ch=2])

        # seems jsense and espirit wants (coil,h,w), can't do (sl, coil, h, w)
        redo_smaps = False
        if redo_smaps:
            smaps = spdevice.xp.zeros(kspace.shape, dtype=kspace.dtype)
            for sl in range(kspace.shape[0]):
                ksp_gpu = sp.to_device(kspace[sl], device=sp.Device(0))
                mps = sp.mri.app.JsenseRecon(ksp_gpu, ksp_calib_width=24, mps_ker_width=16, lamda=0.001,
                                             max_iter=20, max_inner_iter=10,
                                             device=spdevice, show_pbar=True).run()
                smaps[sl] = sp.to_device(mps, spdevice)

        smaps = chan2_complex(spdevice.xp.squeeze(smaps))  # (slice, coil, 768, 396)
        Nslices = smaps.shape[0]

        t_case = time.time()
        optimizer.zero_grad()
        loss_avg = 0.0
        for sl in range(Nslices):
            t_sl = time.time()

            smaps_sl = xp.copy(smaps[sl])  # ndarray on cuda (20, 768, 396), complex64
            kspace_sl = xp.copy(kspace[sl])  # ndarray (20, 768, 396), complex64
            kspaceU_sl = kspace_sl * mask_gpu
            with spdevice:
                A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
                Ah = A.H
                Atruth = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_truth)
                # A ishape (768, 396), oshape (20, 768, 396)
                # Ah ishape (20,768,396), oshape(768,396)

            # Get truth on the fly
            im_sl = sp.to_pytorch(Atruth.H * (kspace_sl * mask_truth), requires_grad=False)

            A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
            Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)

            imEst = 0.0 * sp.to_pytorch(Ah * kspaceU_sl, requires_grad=True)

            t = time.time()
            kspaceU_sl = sp.to_pytorch(kspaceU_sl)  # torch.Size([20, 768, 396, 2])
            imEst2 = ReconModel(imEst, kspaceU_sl, A_torch, Ah_torch)  # (768, 396, 2)

            if WHICH_LOSS == 'mse':
                loss = mseloss_fcn(imEst2, im_sl)
            elif WHICH_LOSS == 'perceptual':
                loss = loss_perceptual(imEst2, im_sl)
            elif WHICH_LOSS == 'patchGAN':
                loss = loss_GAN(imEst2, im_sl, patchGAN)
            else:
                if epoch < epochMSE:
                    loss = mseloss_fcn(imEst2, im_sl)
                    loss_mse_tensor = loss
                else:
                    loss = learnedloss_fcn(imEst2, im_sl, score, rank_trained_on_mag=rank_trained_on_mag)
                    loss_mse_tensor = mseloss_fcn(imEst2, im_sl).detach()

            loss.backward(retain_graph=True)
            # train_avg.update(loss.detach().item(), BATCH_SIZE)
            del imEst2
            if WHICH_LOSS == 'learned':
                with torch.no_grad():
                    loss_learnedT.append(loss.detach().item())
                    loss_mseT.append(float(loss_mse_tensor))

            loss_avg += loss.detach().item()
            loss_avg /= Nslices
            train_avg.update(loss.detach().item())

            del smaps_sl, kspaceU_sl, kspace_sl, im_sl

        # Slice loop
        optimizer.step()
        if epochMSE != 0:
            scheduler.step()

        print(f'case {i}, took {time.time() - t_case}s, loss {loss_avg}, avg = {train_avg.avg()}')

        del imEst
        del smaps, kspace, im
        A = None
        Ah = None
        A_torch = None
        Ah_torch = None
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        torch.cuda.empty_cache()
        # print(f'---------------------after training case {i}, epoch {epoch}-------------------------------')
        # print_mem()

        if i == Ntrain:
            break
    print(f'Epoch took {time.time() - tstart_batch}')

    ReconModel.eval()
    for i, data in enumerate(loader_V, 0):
        smaps, im, kspace = data

        im = torch.squeeze(im, dim=0)

        smaps = sp.to_device(smaps, device=spdevice)
        smaps = spdevice.xp.squeeze(smaps)
        smaps = chan2_complex(spdevice.xp.squeeze(smaps))
        Nslices = smaps.shape[0]

        kspace *= 1e6
        kspace = sp.to_device(kspace, device=spdevice)
        kspace = spdevice.xp.squeeze(kspace)
        kspace = chan2_complex(kspace)

        for sl in range(smaps.shape[0]):
            smaps_sl = xp.copy(smaps[sl])
            kspace_sl = xp.copy(kspace[sl])
            kspaceU_sl = kspace_sl * mask_gpu

            with spdevice:
                A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
                Ah = A.H
                Atruth = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_truth)

            # Get truth
            im_sl = sp.to_pytorch(Atruth.H * (kspace_sl * mask_truth), requires_grad=False)

            A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
            Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)

            imEst = 0.0 * sp.to_pytorch(Ah * kspaceU_sl, requires_grad=False)

            t = time.time()
            kspaceU_sl = sp.to_pytorch(kspaceU_sl)
            imEst2 = ReconModel(imEst, kspaceU_sl, A_torch, Ah_torch)

            if WHICH_LOSS == 'mse':
                loss = mseloss_fcn(imEst2, im_sl)
            elif WHICH_LOSS == 'perceptual':
                loss = loss_perceptual(imEst2, im_sl)
            elif WHICH_LOSS == 'patchGAN':
                loss = loss_GAN(imEst2, im_sl, patchGAN)
            else:
                if epoch < epochMSE:
                    loss = mseloss_fcn(imEst2, im_sl)
                    loss_mse_tensor = loss
                else:
                    loss_mse_tensor = mseloss_fcn(imEst2, im_sl).detach()
                    loss = learnedloss_fcn(imEst2, im_sl, score, rank_trained_on_mag=rank_trained_on_mag)

            eval_avg.update(loss.detach().item(), n=BATCH_SIZE)

            if WHICH_LOSS == 'learned':
                with torch.no_grad():
                    loss_learnedV.append(loss.detach().item())
                    loss_mseV.append(float(loss_mse_tensor))

            if i == 1 and sl == 4:
                truthplt = chan2_complex(torch.squeeze(im_sl.detach().cpu()))
                perturbed = Ah_torch.apply(kspaceU_sl)
                noisyplt = chan2_complex(perturbed.detach().cpu())
                del perturbed
                temp = imEst2
                temp = temp.detach().cpu()
                imEstplt = chan2_complex(temp)
                del temp
                imEstfig = plt_recon(torch.abs(imEstplt))
                writer_val.add_figure('Recon_val', imEstfig, epoch)

                # SENSE
                if epoch == 0:
                    kspaceU_sl_gpu = sp.from_pytorch(kspaceU_sl, iscomplex=True)
                    imSense = sp.mri.app.SenseRecon(kspaceU_sl_gpu, smaps_sl, weights=mask_gpu, lamda=.01,
                                                    max_iter=20, device=spdevice).run()
                    # L1-wavelet
                    imL1 = sp.mri.app.L1WaveletRecon(kspaceU_sl_gpu, smaps_sl, weights=mask_gpu, lamda=.001,
                                                     max_iter=20, device=spdevice).run()

                with h5py.File(out_name, 'a') as hf:
                    if epoch == 0:
                        hf.create_dataset(f"{epoch}_truth", data=np.abs(truthplt.numpy()))
                        hf.create_dataset(f"{epoch}_FT", data=np.abs(noisyplt.numpy()))
                        hf.create_dataset(f"{epoch}_Sense", data=np.abs(imSense.get()))
                        hf.create_dataset(f"{epoch}_L1", data=np.abs(imL1.get()))
                        hf.create_dataset(f"p{epoch}_truth", data=np.angle(truthplt.numpy()))
                        hf.create_dataset(f"p{epoch}_FT", data=np.angle(noisyplt.numpy()))
                        hf.create_dataset(f"p{epoch}_Sense", data=np.angle(imSense.get()))
                        hf.create_dataset(f"p{epoch}_L1", data=np.angle(imL1.get()))

                    hf.create_dataset(f"p{epoch}_recon", data=np.angle(imEstplt.numpy()))
                    hf.create_dataset(f"{epoch}_recon", data=np.abs(imEstplt.numpy()))
                del truthplt, noisyplt, imEstplt

            del smaps_sl, kspaceU_sl, im_sl
        del imEst
        del smaps, im, kspace
        A = None
        Ah = None
        A_torch = None
        Ah_torch = None
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        torch.cuda.empty_cache()

        if i == Nval:
            break

    logging.info(
        f'epoch {epoch} took {(time.time() - tt) / 60} min, Loss train = {train_avg.avg()}, Loss eval = {eval_avg.avg()}')

    # torch.cuda.empty_cache()

    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)

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

    # save models
    state = {
        'state_dict': ReconModel.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss_train': lossT,
        'loss_cal': lossV
    }
    torch.save(state, os.path.join(log_dir, f'Recon{Ntrial}_{WHICH_LOSS}.pt'))
