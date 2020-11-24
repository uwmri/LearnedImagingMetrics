import cupy
import h5py as h5
import pickle
import matplotlib.pyplot as plt
import csv
import logging
import time

import torch
import torchvision
import torch.optim as optim
import torchsummary
from torch.utils.tensorboard import SummaryWriter
try:
    from ax.service.managed_loop import optimize
except:
    print('Can not load ax, botorch not supported')

import sigpy as sp
import sigpy.mri as mri

# from fastMRI.data import transforms as T

from utils.Recon_helper import *
from utils.CreateImagePairs import find, get_smaps, get_truth
from utils.utils_DL import *


mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()

spdevice = sp.Device(0)
Ntrial = 0

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
        parser.add_argument('--pname', type=str, default=f'chenwei_metrics_{Ntrial}')
        args = parser.parse_args()

        setproctitle.setproctitle(args.pname)
        print(f'Setting program name to {args.pname}')
    except:
        print('setproctitle not installled,unavailable, or failed')


else:
    filepath_rankModel = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
    filepath_train = Path("I:/NYUbrain")
    filepath_val = Path("I:/NYUbrain")

log_dir = filepath_rankModel
rank_channel =1
rank_trained_on_mag = True
BO = False
logging.basicConfig(filename=os.path.join(log_dir,f'Recon_{Ntrial}_{DGX}.log'), filemode='w', level=logging.INFO)

#file_rankModel = os.path.join(filepath_rankModel, "RankClassifier16.pt")
file_rankModel = os.path.join(filepath_rankModel, "RankClassifier2902_pretrained.pt")

os.chdir(filepath_rankModel)

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

acc = 8
logging.info(f'Acceleration = {acc}')
# fixed sampling mask
WHICH_MASK='poisson'
if WHICH_MASK=='poisson':
    mask = mri.poisson((xres, yres), accel=acc, crop_corner=True, return_density=False, dtype='float32')
elif WHICH_MASK=='randLines':
    length_center = 20
    num_peri = int((xres*yres/acc-xres*length_center)/xres)
    acquired_center = np.arange((yres-2)/2-(length_center/2-1),yres/2+length_center/2, step=1, dtype='int')
    acquired_peri = np.random.randint(0,(yres-1), num_peri, dtype='int')
    mask = np.zeros((xres, yres), dtype=np.float32)
    mask[:,acquired_center] = 1
    mask[:,acquired_peri] = 1
else:
    mask = np.ones((xres, yres), dtype=np.float32)
mask_gpu = sp.to_device(mask, spdevice)

# Data generator
BATCH_SIZE = 1
prefetch_data = False
logging.info(f'Load train data from {filepath_train}')
trainingset = DataGeneratorRecon(filepath_train, file_train, mask, rank_trained_on_mag=True,data_type=smap_type)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

logging.info(f'Load eval data from {filepath_val}')
validationset = DataGeneratorRecon(filepath_val, file_val, mask, rank_trained_on_mag=True, data_type=smap_type)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=False)

# # check a training dataset
# sneakpeek(trainingset, rank_trained_on_mag)
# sneakpeek(validationset, rank_trained_on_mag)

imSlShape = (BATCH_SIZE,) + (xres, yres)    # (1, 768, 396)

UNROLL = True
ReconModel = MoDL()
ReconModel.cuda()

# torchsummary.summary(ReconModel.denoiser, input_size=(2,768,396), batch_size=16)


# for BO
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_evaluate_recon(parameterization):

    net = ReconModel
    net = train_modRecon(net=net, train_loader=loader_T, mask_gpu=mask_gpu, parameters=parameterization, dtype=torch.float, device=device)
    return evaluate_modRecon(
        net=net,
        data_loader=loader_V,
        mask_gpu=mask_gpu,
        dtype=torch.float,
        device=device
    )


# training
writer_train = SummaryWriter(f'runs/recon/train_{Ntrial}')
writer_val = SummaryWriter(f'runs/recon/val_{Ntrial}')

WHICH_LOSS = 'learned'
OneNet = False
if WHICH_LOSS == 'perceptual':
    loss_perceptual = PerceptualLoss_VGG16()
    loss_perceptual.cuda()
elif WHICH_LOSS == 'patchGAN':
    patchGAN = NLayerDiscriminator(input_nc=2)
    patchGAN.cuda()

INNER_ITER = 10

Nepoch = 50
epochMSE = 0
logging.info(f'MSE for first {epochMSE} epochs then switch to learned')
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)

Ntrain = 252
Nval = 28

# save some images during training
out_name = os.path.join(log_dir,f'sneakpeek_training{Ntrial}_KMJ.h5')
try:
    os.remove(out_name)
except OSError:
    pass

if BO:
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-4, 1], "log_scale": True},
        ],
        evaluation_function=train_evaluate_recon,
        objective_name='mse',
    )

    optimizer = optim.Adam(ReconModel.parameters(), lr=best_parameters['lr'])

    print(best_parameters)
    logging.info(f'BO, {best_parameters}')

else:
    LR = 1e-6
    #optimizer = optim.SGD(ReconModel.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.Adam(ReconModel.parameters(), lr=LR)
    logging.info(f'Adam, lr = {LR}')

logging.info('case averaged loss')
logging.info('Denoiser = Unet')
logging.info(f'inner_iter = {INNER_ITER}')
logging.info(f'{Ntrain} cases training, {Nval} cases validation')
logging.info(f'loss = {WHICH_LOSS}')
logging.info(f'acc = {acc}, mask is {WHICH_MASK}')

ifSingleSlice = True
logging.info(f'ifSingleSlice={ifSingleSlice}')

xp = spdevice.xp


# for epoch in range(Nepoch):
#
#     # Setup counter to keep track of loss
#     train_avg = RunningAverage()
#     eval_avg = RunningAverage()
#
#     ReconModel.train()
#     tt = time.time()
#
#     if prefetch_data:
#         prefetcher = DataPrefetcher(loader_T)
#     else:
#         prefetcher = DataIterator(loader_T)
#
#     smaps, im, kspaceU = prefetcher.next()
#
#     # print(f'-------------------------------beginning of training, epoch {epoch}-------------------------------')
#     # print_mem()
#
#     i = 0
#     tstart_batch = time.time()
#     loss_mseT = []
#     loss_learnedT = []
#     loss_mseV = []
#     loss_learnedV = []
#
#     while kspaceU is not None:
#         i += 1
#
#
#         # smaps is cupy/numpy [1, slice, coil, 768, 396]
#         # im is on torch, ([16, 396, 396, 2])
#         # kspaceU is on torch ([16, 20, 768, 396, 2])
#
#
#         # smaps is torch tensor on cpu [1, slice, coil, 768, 396, 2] (both prefetcher and iterator)
#         # im is on cuda, ([16, 396, 396, ch]) or on cuda [1, slice, 768, 396, ch] if DataIterator
#         # kspaceU is on cpu ([1, 16, 20, 768, 396, 2]) or on cpu [1, slice, coil, 768, 396, 2] if DataIterator
#
#         # # Disable memory pool for device memory (GPU)
#         # cupy.cuda.set_allocator(None)
#         #
#         # # Disable memory pool for pinned memory (CPU).
#         # cupy.cuda.set_pinned_memory_allocator(None)
#
#         smaps = sp.to_device(smaps, device=spdevice)
#         kspaceU = sp.to_device(kspaceU, device=spdevice)
#
#         smaps = spdevice.xp.squeeze(smaps)
#         kspaceU = chan2_complex(spdevice.xp.squeeze(kspaceU))      # cupy array on cuda
#         #kspaceU = sp.to_pytorch(kspaceU, requires_grad=False)
#
#         im = torch.squeeze(im, dim=0)   # torch.Size([16, 768, 396, 1])
#
#         # im = im.cuda()
#         # kspaceU = xp.copy(kspaceU)
#         # smaps = xp.copy(smaps)
#
#         # seems jsense and espirit wants (coil,h,w), can't do (sl, coil, h, w)
#         redo_smaps = False
#         if redo_smaps:
#             smaps = spdevice.xp.zeros(kspaceU.shape, dtype=kspaceU.dtype)
#             for sl in range(kspaceU.shape[0]):
#                 ksp_gpu = sp.to_device(kspaceU[sl], device=sp.Device(0))
#                 mps = sp.mri.app.JsenseRecon(ksp_gpu, ksp_calib_width=24, mps_ker_width=16, lamda=0.001,
#                                              max_iter=20, max_inner_iter=10,
#                                              device=spdevice, show_pbar=True).run()
#                 smaps[sl] = sp.to_device(mps, spdevice)
#
#         smaps = chan2_complex(spdevice.xp.squeeze(smaps))  # (slice, coil, 768, 396)
#         Nslices = smaps.shape[0]
#         #print(f'Load batch {time.time()-t}, {Nslices} {smaps.shape}')
#
#         t = time.time()
#         if ifSingleSlice:
#             loss_temp = 0
#             for sl in range(Nslices):
#                 t_sl = time.time()
#                 smaps_sl = xp.copy(smaps[sl])                              # ndarray on cuda (20, 768, 396), complex64
#                 im_sl = im[sl].clone().cuda()                                               # tensor on cuda (768, 396, 2)
#                 kspaceU_sl = xp.copy(kspaceU[sl])                            # ndarray (20, 768, 396)
#                 with spdevice:
#                     A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
#                     Ah = A.H
#                     # A ishape (768, 396), oshape (20, 768, 396)
#                     # Ah ishape (20,768,396), oshape(768,396)
#
#                 A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
#                 Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)
#
#                 imEst = 0.0 * sp.to_pytorch(Ah * kspaceU_sl, requires_grad=False)
#
#                 t = time.time()
#                 kspaceU_sl = sp.to_pytorch(kspaceU_sl)
#                 optimizer.zero_grad()
#
#                 for inner_iter in range(INNER_ITER):
#
#                     imEst2 = ReconModel(imEst, kspaceU_sl, A_torch, Ah_torch)    # (768, 396, 2)
#
#
#                     # # scale to ~same as truth
#                     # y_pred_real = imEst2[..., 0].detach()
#                     # y_pred_imag = imEst2[..., 1].detach()
#                     # scale_real = torch.sum(torch.transpose(y_pred_real, 0, 1) @ im_sl[..., 0].detach()) / \
#                     #              torch.sum(torch.transpose(y_pred_real, 0, 1) @ y_pred_real)
#                     # scale_imag = torch.sum(torch.transpose(-y_pred_imag, 0, 1) @ im_sl[..., 0].detach()) / \
#                     #              torch.sum(torch.transpose(-y_pred_imag, 0, 1) @ y_pred_real)
#                     # # print(scale_imag.requires_grad)
#                     # # print(f'scale real = {scale_real}, scale_imag = {scale_imag}')
#                     # imEst2[..., 0] *= scale_real
#                     # imEst2[..., 1] *= scale_imag
#                     # del y_pred_real, y_pred_imag
#
#
#                     if WHICH_LOSS == 'mse':
#                         loss_temp = mseloss_fcn(imEst2, im_sl)
#                     elif WHICH_LOSS == 'perceptual':
#                         loss_temp = loss_perceptual(imEst2, im_sl)
#                     elif WHICH_LOSS == 'patchGAN':
#                         loss_temp= loss_GAN(imEst2, im_sl, patchGAN)
#                     else:
#                         if epoch < epochMSE:
#                             loss_temp = mseloss_fcn(imEst2, im_sl)
#                         else:
#                             loss_temp = learnedloss_fcn(imEst2, im_sl, score)
#                             loss_mse_tensor = mseloss_fcn(imEst2, im_sl).detach()
#
#                     loss = loss_temp
#                     del loss_temp
#
#                     if inner_iter == INNER_ITER-1:
#                         loss.backward()
#                     else:
#                         loss.backward(retain_graph=True)
#                     # train_avg.update(loss.detach().item(), BATCH_SIZE)
#                     imEst = imEst2
#
#                     del imEst2
#                 if WHICH_LOSS == 'learned':
#                     with torch.no_grad():
#                         loss_learnedT.append(loss.detach().item())
#                         loss_mseT.append(float(loss_mse_tensor))
#
#
#                 # y_pred_max, _ = torch.max(imEst.detach(), dim=0, keepdim=True)
#                 # y_pred_max, _ = torch.max(y_pred_max, dim=1, keepdim=True)
#                 # truth_max, _ = torch.max(im_sl.detach(), dim=0, keepdim=True)
#                 # truth_max, _ = torch.max(truth_max, dim=1, keepdim=True)
#                 # y_pred_min, _ = torch.min(imEst.detach(), dim=0, keepdim=True)
#                 # y_pred_min, _ = torch.min(y_pred_min, dim=1, keepdim=True)
#                 # truth_min, _ = torch.min(im_sl.detach(), dim=0, keepdim=True)
#                 # truth_min, _ = torch.min(truth_min, dim=1, keepdim=True)
#                 #
#                 # print(f'y_pred_real max = {y_pred_max[0, 0, 0]}, y_pred_imag max = {y_pred_max[0, 0, 1]}')
#                 # print(f'truth_real max = {truth_max[0, 0, 0]}, truth_imag max = {truth_max[0, 0, 1]}')
#                 # print(f'y_pred_real min = {y_pred_min[0, 0, 0]}, y_pred_imag min = {y_pred_min[0, 0, 1]}')
#                 # print(f'truth_real min = {truth_min[0, 0, 0]}, truth_imag min = {truth_min[0, 0, 1]}')
#                 # del y_pred_max, y_pred_min, truth_max, truth_min
#
#                 print(f'case {i}, slice {sl}, took {time.time() - t_sl}s')
#
#                 optimizer.step()
#
#                 del smaps_sl, kspaceU_sl, im_sl
#
#             # case average loss
#             #     loss_temp += loss_temp
#             # loss_temp /= Nslices
#             # loss = loss_temp
#             # loss.backward(retain_graph=True)
#
#             # optimizer.step()
#
#
#         else:
#             A = sp.linop.Diag(
#                 [sp.mri.linop.Sense(smaps[s, :, :, :], coil_batch_size=None) for s in range(Nslices)])
#             Rs1 = sp.linop.Reshape(oshape=A.ishape, ishape=(Nslices, xres, yres))
#             Rs2 = sp.linop.Reshape(oshape=(Nslices, Ncoils, xres, yres), ishape=A.oshape)
#
#             SENSE = Rs2 * A * Rs1
#             SENSEH = SENSE.H
#
#             # AhA = Ah * A
#             # max_eigen = sp.app.MaxEig(AhA, dtype=smaps_sl.dtype, device=spdevice, max_iter=30).run()
#
#             SENSE_torch = sp.to_pytorch_function(SENSE, input_iscomplex=True, output_iscomplex=True)
#             SENSEH_torch = sp.to_pytorch_function(SENSEH, input_iscomplex=True, output_iscomplex=True)
#
#
#             imEst = 0.0 * sp.to_pytorch(SENSEH * kspaceU, requires_grad=False)
#             im_iter = []
#             im_grad = []
#
#
#             kspaceU = sp.to_pytorch(kspaceU, requires_grad=False)
#
#
#             t = time.time()
#             optimizer.zero_grad()
#             for inner_iter in range(INNER_ITER):
#                 imEst2 = ReconModel(imEst, kspaceU, SENSE_torch, SENSEH_torch)
#                 if WHICH_LOSS == 'mse':
#                     loss = mseloss_fcn(imEst2, im)
#                     print(f'mse loss of batch {i} at inner_iter{inner_iter}, epoch{epoch} is {loss} ')
#                 elif WHICH_LOSS == 'perceptual':
#                     loss = loss_perceptual(imEst2, im)
#                 elif WHICH_LOSS == 'patchGAN':
#                     loss = loss_GAN(imEst2, im, patchGAN)
#                 else:
#                     loss = learnedloss_fcn(imEst2, im, score)
#                     torch.cuda.empty_cache()
#
#                 loss.backward(retain_graph=True)
#                 train_avg.update(loss.item(), n=BATCH_SIZE)
#
#                 imEst = imEst2
#                 im_iter.append( imEst.cpu().detach().numpy())
#                 im_iter.append( imEst.cpu().detach().numpy())
#
#             check_loop = False
#             if check_loop:
#                 imS = SENSEH*kspaceU
#                 imS = sp.to_device(imS, sp.cpu_device)
#                 plt.figure()
#                 plt.imshow(np.abs(imS[2]))
#                 plt.show()
#
#                 for imp, img in zip(im_iter, im_grad):
#                     plt.figure()
#                     plt.subplot(121)
#                     plt.imshow(np.abs(chan2_complex(imp[2])))
#                     plt.subplot(122)
#                     plt.imshow(np.abs(chan2_complex(img[2])))
#                     plt.show()
#                 exit()
#
#
#             print(f'{WHICH_LOSS} loss of batch {i}, epoch{epoch} is {loss} ')
#             optimizer.step()
#
#
#             if epoch%10 == 0:
#                 logging.info(f'Inner iteration took {time.time()-t}s')
#
#
#         train_avg.update(loss.detach().item(), n=BATCH_SIZE)
#
#
#
#         if i % 10 ==0 :
#             logging.info(f'Training Loss for batch {i} = {loss.detach().item()}')
#
#         if i == Ntrain:
#             break
#         #
#         # #del smaps,im, kspaceU, imEst2
#         # #mempool.free_all_blocks()
#         # #pinned_mempool.free_all_blocks()
#         # #torch.cuda.empty_cache()
#         # # print(f'Total time = {time.time() - tt}')
#         # #exit()
#         del imEst
#         del smaps, kspaceU, im
#         A = None
#         Ah = None
#         A_torch = None
#         Ah_torch = None
#         mempool.free_all_blocks()
#         pinned_mempool.free_all_blocks()
#         torch.cuda.empty_cache()
#         smaps, im, kspaceU = prefetcher.next()
#
#         # print(f'---------------------after training case {i}, epoch {epoch}-------------------------------')
#         # print_mem()
#
#     print(f'Epoch took {time.time() - tstart_batch}')
#
#     # Validation
#     # with torch.no_grad():
#     ReconModel.eval()
#     if prefetch_data:
#         prefetcher = DataPrefetcher(loader_V)
#     else:
#         prefetcher = DataIterator(loader_V)
#
#     smaps, im, kspaceU = prefetcher.next()
#     i = 0
#
#     # print(f'-------------------------------beginning of eval, epoch {epoch}-------------------------------')
#     # print_mem()
#
#     while kspaceU is not None:
#         i += 1
#         im = torch.squeeze(im, dim=0)
#
#         # # Disable memory pool for device memory (GPU)
#         # cupy.cuda.set_allocator(None)
#         #
#         # # Disable memory pool for pinned memory (CPU).
#         # cupy.cuda.set_pinned_memory_allocator(None)
#
#         smaps = sp.to_device(smaps, device=spdevice)
#         kspaceU = sp.to_device(kspaceU, device=spdevice)
#
#         smaps = spdevice.xp.squeeze(smaps)
#         kspaceU = spdevice.xp.squeeze(kspaceU)
#         kspaceU = chan2_complex(kspaceU)
#
#         smaps = chan2_complex(spdevice.xp.squeeze(smaps))
#         Nslices = smaps.shape[0]
#
#         #im = im.clone().cuda()
#         # kspaceU = xp.copy(kspaceU)
#         # smaps = xp.copy(smaps)
#
#         if ifSingleSlice:
#             loss_temp = 0
#             for sl in range(smaps.shape[0]):
#                 smaps_sl = xp.copy(smaps[sl])
#                 im_sl = im[sl].clone().cuda()
#                 kspaceU_sl = xp.copy(kspaceU[sl])
#                 with spdevice:
#                     A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
#                     Ah = A.H
#
#                 A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
#                 Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)
#
#
#                 imEst = 0.0 * sp.to_pytorch(Ah * kspaceU_sl, requires_grad=False)
#
#                 t = time.time()
#                 kspaceU_sl = sp.to_pytorch(kspaceU_sl)
#                 for inner_iter in range(INNER_ITER):
#                     imEst2 = ReconModel(imEst, kspaceU_sl, A_torch, Ah_torch)
#
#                     # # scale to ~same as truth
#                     # y_pred_real = imEst2[..., 0].detach()
#                     # y_pred_imag = imEst2[..., 1].detach()
#                     # scale_real = torch.sum(torch.transpose(y_pred_real, 0, 1) @ im_sl[..., 0].detach()) / \
#                     #              torch.sum(torch.transpose(y_pred_real, 0, 1) @ y_pred_real)
#                     # scale_imag = torch.sum(torch.transpose(-y_pred_imag, 0, 1) @ im_sl[..., 0].detach()) / \
#                     #              torch.sum(torch.transpose(-y_pred_imag, 0, 1) @ y_pred_real)
#                     # # print(scale_imag.requires_grad)
#                     # # print(f'scale real = {scale_real}, scale_imag = {scale_imag}')
#                     # imEst2[..., 0] *= scale_real
#                     # imEst2[..., 1] *= scale_imag
#                     # del y_pred_real, y_pred_imag
#                     #
#                     # y_pred_max, _ = torch.max(imEst.detach(), dim=0, keepdim=True)
#                     # y_pred_max, _ = torch.max(y_pred_max, dim=1, keepdim=True)
#                     # truth_max, _ = torch.max(im_sl.detach(), dim=0, keepdim=True)
#                     # truth_max, _ = torch.max(truth_max, dim=1, keepdim=True)
#                     # y_pred_min, _ = torch.min(imEst.detach(), dim=0, keepdim=True)
#                     # y_pred_min, _ = torch.min(y_pred_min, dim=1, keepdim=True)
#                     # truth_min, _ = torch.min(im_sl.detach(), dim=0, keepdim=True)
#                     # truth_min, _ = torch.min(truth_min, dim=1, keepdim=True)
#                     #
#                     # print(f'y_pred_real max = {y_pred_max[0, 0, 0]}, y_pred_imag max = {y_pred_max[0, 0, 1]}')
#                     # print(f'truth_real max = {truth_max[0, 0, 0]}, truth_imag max = {truth_max[0, 0, 1]}')
#                     # print(f'y_pred_real min = {y_pred_min[0, 0, 0]}, y_pred_imag min = {y_pred_min[0, 0, 1]}')
#                     # print(f'truth_real min = {truth_min[0, 0, 0]}, truth_imag min = {truth_min[0, 0, 1]}')
#                     # del y_pred_max, y_pred_min, truth_max, truth_min
#
#
#                     if WHICH_LOSS == 'mse':
#                         loss_temp = mseloss_fcn(imEst2, im_sl)
#                     elif WHICH_LOSS == 'perceptual':
#                         loss_temp = loss_perceptual(imEst2, im_sl)
#                     elif WHICH_LOSS == 'patchGAN':
#                         loss_temp = loss_GAN(imEst2, im_sl, patchGAN)
#                     else:
#                         if epoch < epochMSE:
#                             loss_temp = mseloss_fcn(imEst2, im_sl)
#                         else:
#                             loss_mse_tensor = mseloss_fcn(imEst2, im_sl).detach()
#                             loss_temp = learnedloss_fcn(imEst2, im_sl, score)
#                     loss = loss_temp
#                     imEst = imEst2
#
#                     del imEst2, loss_temp
#
#                 eval_avg.update(loss.detach().item(), n=BATCH_SIZE)
#
#                 if WHICH_LOSS == 'learned':
#                     with torch.no_grad():
#                         loss_learnedV.append(loss.detach().item())
#                         loss_mseV.append(float(loss_mse_tensor))
#
#
#                 if i == 1 and sl == 4:
#                     truthplt = torch.squeeze(im_sl.cpu())
#                     perturbed = Ah_torch.apply(kspaceU_sl)
#                     noisyplt = torch.abs(chan2_complex(perturbed.detach().cpu()))
#                     del perturbed
#                     temp = imEst
#                     temp = temp.detach().cpu()
#                     imEstplt = torch.abs(chan2_complex(temp))
#                     del temp
#                     # truthplt = torch.unsqueeze(truthplt, 0)
#                     # noisyplt = torch.unsqueeze(noisyplt, 0)
#                     # imEstplt = torch.unsqueeze(imEstplt, 0)
#                     imEstfig = plt_recon(imEstplt)
#                     writer_val.add_figure('Recon_val', imEstfig, epoch)
#
#                     with h5py.File(out_name, 'a') as hf:
#                         if epoch == 0:
#                             hf.create_dataset(f"{epoch}_truth", data=truthplt.numpy())
#                             hf.create_dataset(f"{epoch}_FT", data=noisyplt.numpy())
#                             hf.create_dataset(f"{epoch}_recon", data=imEstplt.numpy())
#                         else:
#                             hf.create_dataset(f"{epoch}_recon", data=imEstplt.numpy())
#                     del truthplt, noisyplt, imEstplt
#
#                 del smaps_sl, kspaceU_sl, im_sl
#
#             # loss_temp /= Nslices
#             # loss = loss_temp
#             # eval_avg.update(loss.detach().item(), n=BATCH_SIZE)
#
#             if i == Nval:
#                 break
#
#
#
#         else:
#             A = sp.linop.Diag(
#                 [sp.mri.linop.Sense(smaps[s, :, :, :], weights=mask_gpu, coil_batch_size=None) for s in range(Nslices)])
#             Rs1 = sp.linop.Reshape(oshape=A.ishape, ishape=(Nslices, xres, yres))
#             Rs2 = sp.linop.Reshape(oshape=(Nslices, Ncoils, xres, yres), ishape=A.oshape)
#
#             SENSE = Rs2 * A * Rs1
#             SENSEH = SENSE.H
#
#             # AhA = Ah * A
#             # max_eigen = sp.app.MaxEig(AhA, dtype=smaps_sl.dtype, device=spdevice, max_iter=30).run()
#
#             SENSE_torch = sp.to_pytorch_function(SENSE, input_iscomplex=True, output_iscomplex=True)
#             SENSEH_torch = sp.to_pytorch_function(SENSEH, input_iscomplex=True, output_iscomplex=True)
#
#             # Initial guess
#             #imEst = 0 * SENSEH_torch.apply(kspaceU)
#
#             imEst = 0.0 * sp.to_pytorch(SENSEH * kspaceU, requires_grad=False)
#
#             kspaceU = sp.to_pytorch(kspaceU, requires_grad=False)
#
#             # forward
#             for inner_iter in range(INNER_ITER):
#                 imEst2 = ReconModel(imEst, kspaceU, SENSE_torch, SENSEH_torch)
#
#                 if WHICH_LOSS == 'mse':
#                     loss = mseloss_fcn(imEst2, im)
#                 elif WHICH_LOSS == 'perceptual':
#                     loss = loss_perceptual(imEst2, im)
#                 elif WHICH_LOSS == 'patchGAN':
#                     loss = loss_GAN(imEst2, im, patchGAN)
#                 else:
#                     loss = learnedloss_fcn(imEst2, im, score)
#                     torch.cuda.empty_cache()
#
#                 imEst = imEst2
#                 del imEst2
#             loss /= INNER_ITER
#             print(f'{WHICH_LOSS} loss of batch {i} at inner_iter{inner_iter}, epoch{epoch} is {loss} ')
#
#             eval_avg.update(loss.item(), n=BATCH_SIZE)
#
#             if i == 1:
#                 truthplt = torch.abs(torch.squeeze(chan2_complex(im.cpu())))
#                 perturbed = SENSEH_torch.apply(kspaceU)
#                 noisyplt = torch.abs(torch.squeeze(chan2_complex(perturbed.cpu())))
#
#                 temp = imEst
#                 temp = temp.detach().cpu()
#                 imEstplt = torch.abs(torch.squeeze(chan2_complex(temp)))
#                 writer_train.add_image('training', imEstplt[2], 0, dataformats='HW')
#
#                 with h5py.File(out_name, 'a') as hf:
#                     if epoch == 0:
#                         hf.create_dataset(f"{epoch}_truth", data=torch.unsqueeze(truthplt[2], 0).numpy())
#                         hf.create_dataset(f"{epoch}_FT", data=torch.unsqueeze(noisyplt[2], 0).numpy())
#                         hf.create_dataset(f"{epoch}_recon", data=torch.unsqueeze(imEstplt[2],0).numpy())
#                     else:
#                         hf.create_dataset(f"{epoch}_recon", data=torch.unsqueeze(imEstplt[2], 0).numpy())
#
#                 break
#         del imEst
#         del smaps, im, kspaceU
#         A = None
#         Ah = None
#         A_torch = None
#         Ah_torch = None
#         mempool.free_all_blocks()
#         pinned_mempool.free_all_blocks()
#         torch.cuda.empty_cache()
#
#         smaps, im, kspaceU = prefetcher.next()
#
#         # print(f'--------------------------------after val case {i}, epoch {epoch}------------------------------------')
#         # print_mem()
#
#
#     logging.info(f'epoch {epoch} took {(time.time() - tt)/60} min, Loss train = {train_avg.avg()}, Loss eval = {eval_avg.avg()}')
#     #torch.cuda.empty_cache()
#
#     writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
#     writer_train.add_scalar('Loss', train_avg.avg(), epoch)
#
#     logging.info('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
#     lossT[epoch] = train_avg.avg()
#     lossV[epoch] = eval_avg.avg()
#
#     # Score diff vs MSE between images and truths
#     if WHICH_LOSS == 'learned':
#         loss_learnedT = np.array(loss_learnedT)
#         loss_mseT = np.array(loss_mseT)
#         lossplotT = plt_scoreVsMse(loss_learnedT, loss_mseT)
#         writer_train.add_figure('Loss_learned_vs_mse', lossplotT, epoch)
#
#         loss_learnedV = np.array(loss_learnedV)
#         loss_mseV = np.array(loss_mseV)
#         lossplotV = plt_scoreVsMse(loss_learnedV, loss_mseV)
#         writer_val.add_figure('Loss_learned_vs_mse', lossplotV, epoch)


