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
from utils.model_helper import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *
from random import randrange
#from fastmri.models.varnet import *


mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()
spdevice = sp.Device(0)
xp = spdevice.xp

Ntrial = randrange(10000)

# Argument parser
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_folder', type=str,
                    default=r'I:\NYUbrain',
                    help='Data path')
parser.add_argument('--metric_file', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn\RankClassifier5624.pt',
                    help='Name of learned metric file')
parser.add_argument('--log_dir', type=str,
                    default=r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020',

                    help='Directory to log files')
parser.add_argument('--pname', type=str, default=f'chenwei_recon_{Ntrial}')
args = parser.parse_args()

# Set folders
log_dir = args.log_dir
data_folder = args.data_folder
metric_file = args.metric_file

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

ranknet = L2cnn(channels_in=rank_channel, channel_base=2)
#ranknet = ISOResNet2(BasicBlock, [2,2,2,2], for_denoise=False)
classifier = Classifier(ranknet)

state = torch.load(metric_file)
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
mask_torch = sp.to_pytorch(mask_gpu, requires_grad=False)

# Data generator
Ntrain = 20
Nval = 2
BATCH_SIZE = 1
prefetch_data = True
logging.info(f'Load train data from {data_folder}')
trainingset = DataGeneratorRecon(data_folder, file_train, num_cases=Ntrain, rank_trained_on_mag=rank_trained_on_mag,
                                 data_type=smap_type)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

logging.info(f'Load eval data from {data_folder}')
validationset = DataGeneratorRecon(data_folder, file_val, num_cases=Nval, rank_trained_on_mag=rank_trained_on_mag,
                                   data_type=smap_type)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=False)

UNROLL = True
if UNROLL:
    denoiser = 'cnn_shortcut'
    logging.info(f'denoiser is {denoiser}')

    INNER_ITER = 1
    #ReconModel = unrolledK(inner_iter=INNER_ITER, DENOISER=denoiser)
    ReconModel = MoDL(inner_iter=INNER_ITER, DENOISER=denoiser)
    logging.info(f'MoDL, inner iter = {INNER_ITER}')
else:
    NUM_CASCADES = 12
    ReconModel = EEVarNet(num_cascades=NUM_CASCADES)
    logging.info(f'EEVarNet, {NUM_CASCADES} cascades')
ReconModel.cuda();
#summary(ReconModel.denoiser, input_size=(1,768,396))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer_train = SummaryWriter(os.path.join(log_dir,f'runs/recon/train_{Ntrial}'))
writer_val = SummaryWriter(os.path.join(log_dir,f'runs/recon/val_{Ntrial}'))

WHICH_LOSS = 'learned'
if WHICH_LOSS == 'perceptual':
    loss_perceptual = PerceptualLoss_VGG16()
    loss_perceptual.cuda()
elif WHICH_LOSS == 'patchGAN':
    patchGAN = NLayerDiscriminator(input_nc=2)
    patchGAN.cuda()
elif WHICH_LOSS == 'ssim':
    ssim_module = SSIM()

Nepoch = 200
epochMSE = 0
logging.info(f'MSE for first {epochMSE} epochs then switch to learned')
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)

out_name = os.path.join(log_dir,f'Images_training{Ntrial}_{WHICH_LOSS}.h5')
print(f'Logging to {out_name}')

LR = 1e-4
# optimizer = optim.SGD(ReonModel.parameters(), lr=LR, momentum=0.9)
optimizer = optim.Adam(ReconModel.parameters(), lr=LR)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
# if epochMSE != 0:
#     lambda1 = lambda epoch: 1e1 if epoch<epochMSE else 1.0
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

logging.info(f'Adam, lr = {LR}')
logging.info('case averaged loss')
logging.info('Denoiser = cnn_shortcut')
logging.info(f'{Ntrain} cases training, {Nval} cases validation')
logging.info(f'loss = {WHICH_LOSS}')
logging.info(f'acc = {acc}, mask is {WHICH_MASK}')

# # Get the scale for Denoiser
# with torch.no_grad():
#     da = 0.0
#     for avg in range(10):
#         x = torch.randn((1, 2, yres, xres), device='cuda')
#         for iter in range(30):
#             ein = torch.linalg.norm(x)
#             y = ReconModel.call_denoiser(x)
#             eout = torch.linalg.norm(y)
#             x = y / eout
#             # print(f'Scale {eout/ein}')
#         # print(eout)
#         da += eout / 10
#     logging.info(f'Avg Denoiser scale = {da}')
#     ReconModel.set_denoiser_scale( 0.9 / da)

for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    train_avg_mse = RunningAverage()
    eval_avg = RunningAverage()
    eval_avg_mse = RunningAverage()

    loss_mseT = []
    loss_learnedT = []
    loss_mseV = []
    loss_learnedV = []

    ReconModel.train()
    tt = time.time()
    with torch.autograd.set_detect_anomaly(False):
        for i, data in enumerate(loader_T,0):
            # print(f'-------------------------------beginning of training, epoch {epoch}-------------------------------')
            # print_mem()
            tstart_batch = time.time()

            smaps, kspace = data
            smaps = smaps[0]
            kspace = kspace[0]

            # # seems jsense and espirit wants (coil,h,w), can't do (sl, coil, h, w)
            # redo_smaps = False
            # if redo_smaps:
            #     smaps = spdevice.xp.zeros(kspace.shape, dtype=kspace.dtype)
            #     for sl in range(kspace.shape[0]):
            #         ksp_gpu = sp.to_device(kspace[sl], device=sp.Device(0))
            #         mps = sp.mri.app.JsenseRecon(ksp_gpu, ksp_calib_width=24, mps_ker_width=16, lamda=0.001,
            #                                      max_iter=20, max_inner_iter=10,
            #                                      device=spdevice, show_pbar=True).run()
            #         smaps[sl] = sp.to_device(mps, spdevice)

            t_case = time.time()
            optimizer.zero_grad()
            loss_avg = 0.0
            for sl in range(smaps.shape[0]):
                t_sl = time.time()

                # Clone to torch
                smaps_sl = torch.clone(smaps[sl]).cuda()  # ndarray on cuda (20, 768, 396), complex64
                kspace_sl = torch.clone(kspace[sl]).cuda()  # ndarray (20, 768, 396), complex64

                # Move to torch
                kspaceU_sl = kspace_sl * mask_torch

                # Get truth on the fly
                #im_sl = sp.to_pytorch(Atruth.H * (kspace_sl * mask_truth), requires_grad=False)
                im_sl = sense_adjoint(smaps_sl, kspace_sl)
                idxL = int((im_sl.shape[0] - im_sl.shape[1]) / 2)
                idxR = int(idxL + im_sl.shape[1])
                im_sl = im_sl[idxL:idxR, :]

                # Get PyTorch functions
                #A_torch = sp.to_pytorch_function(A)
                #Ah_torch = sp.to_pytorch_function(Ah)
                #imEst = 0.0 * sp.to_pytorch(Ah * kspaceU_sl, requires_grad=True)
                imEst = torch.zeros_like(im_sl)
                t = time.time()
                #kspaceU_sl = sp.to_pytorch(kspaceU_sl)  # torch.Size([20, 768, 396, 2])

                if UNROLL:
                    imEst2 = ReconModel(imEst, kspaceU_sl, smaps_sl, mask_torch)  # (768, 396, 2)
                else:
                    imEst2 = ReconModel(kspaceU_sl, mask_torch, A_torch, Ah_torch, smaps_sl, mask_torch)

                if WHICH_LOSS == 'mse':
                    loss = mseloss_fcn(imEst2, im_sl)
                elif WHICH_LOSS == 'ssim':
                    imEst2_ep = torch.unsqueeze(imEst2, 0)
                    imEst2_ep = imEst2_ep.permute(0,-1,1,2)
                    im_sl_ep = torch.unsqueeze(im_sl, 0)
                    im_sl_ep = im_sl_ep.permute(0, -1, 1, 2)
                    loss = 1 - ssim(imEst2_ep, im_sl_ep)
                elif WHICH_LOSS == 'perceptual':
                    loss = loss_perceptual(imEst2, im_sl)
                elif WHICH_LOSS == 'patchGAN':
                    loss = loss_GAN(imEst2, im_sl, patchGAN)
                else:
                    if epoch < epochMSE:
                        loss = mseloss_fcn(imEst2, im_sl) * 5e2
                        loss_mse_tensor = loss.item()
                    else:
                        loss = 1e-4 * learnedloss_fcn(imEst2, im_sl, score, rank_trained_on_mag=rank_trained_on_mag)
                        loss_mse_tensor = mseloss_fcn(imEst2, im_sl).detach()

                loss.backward(retain_graph=True)

                # train_avg.update(loss.detach().item(), BATCH_SIZE)
                del imEst2
                if WHICH_LOSS == 'learned':
                    with torch.no_grad():
                        loss_learnedT.append(loss.detach().item())
                        loss_mseT.append(loss_mse_tensor.detach().cpu().item())

                loss_avg += loss.detach().item()
                train_avg.update(loss.detach().item())
                if WHICH_LOSS != 'mse':
                    train_avg_mse.update(loss_mse_tensor.detach().cpu().item())

                del smaps_sl, kspaceU_sl, kspace_sl, im_sl

            # Slice loop
            optimizer.step()
            #scheduler.step()

            print(f'case {i}, took {time.time() - t_case}s, loss {loss_avg}, avg = {train_avg.avg()}')

            # del imEst
            # del smaps, kspace, im
            # A = None
            # Ah = None
            # A_torch = None
            # Ah_torch = None
            # mempool.free_all_blocks()
            # pinned_mempool.free_all_blocks()
            # torch.cuda.empty_cache()
            # print(f'---------------------after training case {i}, epoch {epoch}-------------------------------')
            # print_mem()

            # if i == Ntrain:
            #     break

    print(f'Epoch took {time.time() - tstart_batch}')

    ReconModel.eval()
    for i, data in enumerate(loader_V, 0):
        smaps, kspace = data
        smaps = smaps[0]
        kspace = kspace[0]

        Nslices = smaps.shape[0]

        for sl in range(smaps.shape[0]):
            smaps_sl = torch.clone(smaps[sl]).cuda()
            kspace_sl = torch.clone(kspace[sl]).cuda()

            kspaceU_sl = kspace_sl * mask_torch

            # with spdevice:
            #     A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
            #     Ah = A.H
            #     Atruth = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_truth)

            # Get truth
            im_sl = sense_adjoint(smaps_sl, kspace_sl)
            idxL = int((im_sl.shape[0] - im_sl.shape[1]) / 2)
            idxR = int(idxL + im_sl.shape[1])
            im_sl = im_sl[idxL:idxR, :]

            #A_torch = sp.to_pytorch_function(A)
            #Ah_torch = sp.to_pytorch_function(Ah)

            #imEst = 0.0 * sp.to_pytorch(Ah * kspaceU_sl, requires_grad=False)
            imEst = torch.zeros_like(im_sl)
            t = time.time()
            #kspaceU_sl = sp.to_pytorch(kspaceU_sl)
            if UNROLL:
                imEst2 = ReconModel(imEst, kspaceU_sl, smaps_sl, mask_torch)
            else:
                imEst2 = ReconModel(kspaceU_sl, mask_torch, A_torch, Ah_torch)

            if WHICH_LOSS == 'mse':
                loss = mseloss_fcn(imEst2, im_sl)
            elif WHICH_LOSS == 'ssim':
                imEst2_ep = torch.unsqueeze(imEst2, 0)
                imEst2_ep = imEst2_ep.permute(0, -1, 1, 2)
                im_sl_ep = torch.unsqueeze(im_sl, 0)
                im_sl_ep = im_sl_ep.permute(0, -1, 1, 2)
                loss = 1 - ssim(imEst2_ep, im_sl_ep)
            elif WHICH_LOSS == 'perceptual':
                loss = loss_perceptual(imEst2, im_sl)
            elif WHICH_LOSS == 'patchGAN':
                loss = loss_GAN(imEst2, im_sl, patchGAN)
            else:
                if epoch < epochMSE:
                    loss = mseloss_fcn(imEst2, im_sl) *5e2
                    loss_mse_tensor = loss
                else:
                    loss_mse_tensor = mseloss_fcn(imEst2, im_sl).detach()
                    loss = 1e-4 * learnedloss_fcn(imEst2, im_sl, score, rank_trained_on_mag=rank_trained_on_mag)

            eval_avg.update(loss.detach().item(), n=BATCH_SIZE)
            if WHICH_LOSS != 'mse':
                eval_avg_mse.update(loss_mse_tensor.detach().cpu().item())

            if WHICH_LOSS == 'learned':
                with torch.no_grad():
                    loss_learnedV.append(loss.detach().item())
                    loss_mseV.append(loss_mse_tensor.cpu())

            if i == 0 and sl == 4:
                #truthplt = chan2_complex(torch.squeeze(im_sl.detach().cpu()))
                truthplt = im_sl.detach().cpu()
                #perturbed = Ah_torch.apply(kspaceU_sl)
                perturbed = sense_adjoint(smaps_sl, kspaceU_sl)
                #noisyplt = chan2_complex(perturbed.detach().cpu())
                noisyplt = perturbed.detach().cpu()
                noisyplt = noisyplt[idxL:idxR,:]
                del perturbed
                imEstplt = imEst2.detach().cpu()
                imEstfig = plt_recon(torch.squeeze(torch.abs(imEstplt)))
                writer_val.add_figure('Recon_val', imEstfig, epoch)

                # SENSE
                if epoch == 0:
                    kspaceU_sl_gpu = sp.to_device(sp.from_pytorch(kspaceU_sl.cpu()), sp.Device(0))
                    smaps_sl_gpu = sp.to_device(sp.from_pytorch(smaps_sl.cpu()), sp.Device(0))
                    imSense = sp.mri.app.SenseRecon(kspaceU_sl_gpu, smaps_sl_gpu, weights=mask_gpu, lamda=.01,
                                                    max_iter=20, device=spdevice).run()
                    #imSense = imSense[idxL:idxR,:]
                    # L1-wavelet
                    imL1 = sp.mri.app.L1WaveletRecon(kspaceU_sl_gpu, smaps_sl_gpu, weights=mask_gpu, lamda=.001,
                                                     max_iter=20, device=spdevice).run()
                    #imL1 = imL1[idxL:idxR,:]

                with h5py.File(out_name, 'a') as hf:
                    if epoch == 0:
                        hf.create_dataset(f"{epoch}_truth", data=np.squeeze(np.abs(truthplt.numpy())))
                        hf.create_dataset(f"{epoch}_FT", data=np.squeeze(np.abs(noisyplt.numpy())))
                        hf.create_dataset(f"{epoch}_Sense", data=np.squeeze(np.abs(imSense.get())))
                        hf.create_dataset(f"{epoch}_L1", data=np.squeeze(np.abs(imL1.get())))
                        hf.create_dataset(f"p{epoch}_truth", data=np.squeeze(np.angle(truthplt.numpy())))
                        hf.create_dataset(f"p{epoch}_FT", data=np.squeeze(np.angle(noisyplt.numpy())))
                        hf.create_dataset(f"p{epoch}_Sense", data=np.squeeze(np.angle(imSense.get())))
                        hf.create_dataset(f"p{epoch}_L1", data=np.squeeze(np.angle(imL1.get())))

                    hf.create_dataset(f"p{epoch}_recon", data=np.squeeze(np.angle(imEstplt.numpy())))
                    hf.create_dataset(f"{epoch}_recon", data=np.squeeze(np.abs(imEstplt.numpy())))
                del truthplt, noisyplt, imEstplt

            del smaps_sl, kspaceU_sl, im_sl
        #del imEst
        #del smaps, im, kspace
        #A = None
        #Ah = None
        #A_torch = None
        #Ah_torch = None
        #mempool.free_all_blocks()
        #pinned_mempool.free_all_blocks()
        #torch.cuda.empty_cache()

        # if i == Nval:
        #     break

    logging.info(
        f'epoch {epoch} took {(time.time() - tt) / 60:.2f} min, '
        f'train = {train_avg.avg()}, eval = {eval_avg.avg()}, '
        f'mse eval = {eval_avg_mse.avg()}')

    # torch.cuda.empty_cache()

    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)
    writer_val.add_scalar('MSE Loss', eval_avg_mse.avg(), epoch)
    writer_train.add_scalar('MSE Loss', train_avg_mse.avg(), epoch)

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
#
