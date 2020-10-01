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

# load RankNet
DGX = False
if DGX:
    filepath_rankModel = Path('/raid/DGXUserDataRaid/cxt004/NYUbrain')
    filepath_train = Path('/raid/DGXUserDataRaid/cxt004/NYUbrain')
    filepath_val = Path('/raid/DGXUserDataRaid/cxt004/NYUbrain')
else:
    filepath_rankModel = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
    filepath_train = Path("I:/NYUbrain")
    filepath_val = Path("I:/NYUbrain")

    filepath_rankModel = Path('E:\ImagePairs_Pack_04032020')
    filepath_train = Path("E:/")
    filepath_val = Path("E:/")
    log_dir = Path("E:/")

Ntrial = 2.6
logging.basicConfig(filename=os.path.join(log_dir,f'Recon_{Ntrial}.log'), filemode='w', level=logging.INFO)

file_rankModel = os.path.join(filepath_rankModel, "RankClassifier16.pt")
os.chdir(filepath_rankModel)

classifier = Classifier()

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
score = classifier.rank
score.cuda()

scans_train = 'train_20coil.txt'
scans_val = 'val_20coil.txt'

file_train = 'ksp_truths_smaps_train.h5'
file_val = 'ksp_truths_smaps_val.h5'
smap_type = 'smap32'

file_train = 'ksp_truths_smaps_train_lzf.h5'
file_val = 'ksp_truths_smaps_val_lzf.h5'
smap_type = 'smap16'

# Only choose 20-coil data for now
Ncoils = 20
xres = 768
yres = 396
acc = 8

# fixed sampling mask
print(f'Acceleration = {acc}')
mask = mri.poisson((xres, yres), accel=acc, crop_corner=True, return_density=False, dtype='float32')
#mask = np.ones((xres, yres), dtype=np.float32)

# Data generator
BATCH_SIZE = 1
prefetch_data = False
logging.info(f'Load train data from {filepath_train}')
print('D')
trainingset = DataGeneratorRecon(filepath_train, scans_train, file_train, mask, ifLEARNED=False, data_type=smap_type)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

logging.info(f'Load eval data from {filepath_val}')
validationset = DataGeneratorRecon(filepath_val, scans_val, file_val, mask, ifLEARNED=False, data_type=smap_type)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=False)

# # check a training dataset
# sneakpeek(trainingset)
# sneakpeek(validationset)

imSlShape = (BATCH_SIZE,) + (xres, yres)    # (1, 768, 396)

UNROLL = True

logging.info('Using MoDL Recon')
ReconModel = MoDL()
ReconModel.cuda()


# for BO
BO = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_evaluate_recon(parameterization):

    net = ReconModel
    net = train_modRecon(net=net, train_loader=loader_T, parameters=parameterization, dtype=torch.float, device=device)
    return evaluate_mod(
        net=net,
        data_loader=loader_V,
        dtype=torch.float,
        device=device
    )


# training
writer_train = SummaryWriter(f'runs/recon/train_{Ntrial}')
writer_val = SummaryWriter(f'runs/recon/val_{Ntrial}')

WHICH_LOSS = 'mse'
LEARNED = False
OneNet = False
if WHICH_LOSS == 'perceptual':
    loss_perceptual = PerceptualLoss_VGG16()
    loss_perceptual.cuda()
elif WHICH_LOSS == 'patchGAN':
    patchGAN = NLayerDiscriminator(input_nc=2)
    patchGAN.cuda()

INNER_ITER = 10

Nepoch = 50
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)


# save some images during training
out_name = os.path.join(log_dir,f'sneakpeek_training{Ntrial}.h5')
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

else:
    LR = 1e-3
    #optimizer = optim.SGD(ReconModel.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.Adam(ReconModel.parameters(), lr=LR)
    logging.info(f'Adam, lr = {LR}')

logging.info('Denoiser = CNN_short, with 1 blocks')
logging.info(f'inner_iter = {INNER_ITER}')
logging.info('9 cases training, 1 cases validation')
logging.info(f'loss = {WHICH_LOSS}')
logging.info(f'acc = {acc}')

mask_gpu = sp.to_device(mask, spdevice)
for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    ReconModel.train()
    tt = time.time()

    if prefetch_data:
        prefetcher = DataPrefetcher(loader_T)
    else:
        prefetcher = DataIterator(loader_T)

    smaps, im, kspaceU = prefetcher.next()
    i = 0
    tstart_batch = time.time()

    while kspaceU is not None:
        i += 1

        # smaps is torch tensor on cpu [1, slice, coil, 768, 396]
        # im is on cuda, ([16, 396, 396, 2])
        # kspaceU is on cuda ([16, 20, 768, 396, 2])

        smaps = sp.to_device(smaps, device=spdevice)
        kspaceU = sp.to_device(kspaceU, device=spdevice)

        smaps = spdevice.xp.squeeze(smaps)
        kspaceU = spdevice.xp.squeeze(kspaceU)

        # seems jsense and espirit wants (coil,h,w), can't do (sl, coil, h, w)
        redo_smaps = False
        if redo_smaps:
            smaps = spdevice.xp.zeros(kspaceU.shape, dtype=kspaceU.dtype)
            for sl in range(kspaceU.shape[0]):
                ksp_gpu = sp.to_device(kspaceU[sl], device=sp.Device(0))
                mps = sp.mri.app.JsenseRecon(ksp_gpu, ksp_calib_width=24, mps_ker_width=16, lamda=0.001,
                                             max_iter=20, max_inner_iter=10,
                                             device=spdevice, show_pbar=True).run()
                smaps[sl] = sp.to_device(mps, spdevice)

        #smaps = chan2_complex(spdevice.xp.squeeze(smaps))  # (slice, coil, 768, 396)
        Nslices = smaps.shape[0]
        #print(f'Load batch {time.time()-t}, {Nslices} {smaps.shape}')

        #  t = time.time()
        A = sp.linop.Diag(
            [sp.mri.linop.Sense(smaps[s, :, :, :], weights=mask_gpu, coil_batch_size=None) for s in range(Nslices)])
        Rs1 = sp.linop.Reshape(oshape=A.ishape, ishape=(Nslices, xres, yres))
        Rs2 = sp.linop.Reshape(oshape=(Nslices, Ncoils, xres, yres), ishape=A.oshape)

        SENSE = Rs2 * A * Rs1
        SENSEH = SENSE.H

        SENSE_torch = sp.to_pytorch_function(SENSE, input_iscomplex=True, output_iscomplex=True)
        SENSEH_torch = sp.to_pytorch_function(SENSEH, input_iscomplex=True, output_iscomplex=True)

        imEst = 0.0*sp.to_pytorch(SENSEH*kspaceU, requires_grad=False)

        im_iter = []
        im_grad = []

        kspaceU = sp.to_pytorch(kspaceU)

        t = time.time()
        optimizer.zero_grad()
        for inner_iter in range(INNER_ITER):
            imEst2 = ReconModel(imEst, kspaceU, SENSE_torch, SENSEH_torch, ifLEARNED=LEARNED)
            #im_grad.append( grad_op_torch.apply(imEst).cpu().detach().numpy())

            if WHICH_LOSS == 'mse':
                loss = 0
                loss = mseloss_fcn(imEst2, im) / INNER_ITER
            elif WHICH_LOSS == 'perceptual':
                loss = loss_perceptual(imEst2, im) / INNER_ITER
            elif WHICH_LOSS == 'patchGAN':
                loss = loss_GAN(imEst2, im, patchGAN) / INNER_ITER
            else:
                loss = 0
                for sl in range(imEst2.shape[0]):
                    loss += learnedloss_fcn(imEst2[sl], im[sl], score) / INNER_ITER
                    torch.cuda.empty_cache()
                loss /= imEst2.shape[0]     # slices average loss
                # undo cropping, imEst2 is square while ReconModel input imEst need to be rectangular
                imEst2 = zero_pad_imEst(imEst2)
            #print(f'{inner_iter} loss = {loss}')

            loss.backward(retain_graph=True)
            imEst = imEst2

            im_iter.append( imEst.cpu().detach().numpy())
            im_iter.append( imEst.cpu().detach().numpy())

        check_loop = False
        if check_loop:
            imS = SENSEH*kspaceU
            imS = sp.to_device(imS, sp.cpu_device)
            plt.figure()
            plt.imshow(np.abs(imS[2]))
            plt.show()

            for imp, img in zip(im_iter, im_grad):
                plt.figure()
                plt.subplot(121)
                plt.imshow(np.abs(chan2_complex(imp[2])))
                plt.subplot(122)
                plt.imshow(np.abs(chan2_complex(img[2])))
                plt.show()
            exit()

        print(f'{WHICH_LOSS} loss of batch {i}, epoch{epoch} is {loss} ')
        optimizer.step()

        if epoch%10 == 0:
            logging.info(f'Inner iteration took {time.time()-t}s')

        train_avg.update(loss.item(), n=BATCH_SIZE)
        logging.info(f'Training Loss for batch {i} = {loss.item()}')

        if i == 9:
            break

        #del smaps,im, kspaceU, imEst2
        #mempool.free_all_blocks()
        #pinned_mempool.free_all_blocks()
        #torch.cuda.empty_cache()
        # print(f'Total time = {time.time() - tt}')
        #exit()

        smaps, im, kspaceU = prefetcher.next()

    print(f'Epoch took {time.time() - tstart_batch}')

    # Validation
    # with torch.no_grad():
    ReconModel.eval()
    if prefetch_data:
        prefetcher = DataPrefetcher(loader_V)
    else:
        prefetcher = DataIterator(loader_V)

    smaps, im, kspaceU = prefetcher.next()
    i = 0
    while kspaceU is not None:
        i += 1

        smaps = sp.to_device(smaps, device=spdevice)
        kspaceU = sp.to_device(kspaceU, device=spdevice)

        smaps = spdevice.xp.squeeze(smaps)
        kspaceU = spdevice.xp.squeeze(kspaceU)

        Nslices = smaps.shape[0]

        #  t = time.time()
        A = sp.linop.Diag(
            [sp.mri.linop.Sense(smaps[s, :, :, :], weights=mask_gpu, coil_batch_size=None) for s in range(Nslices)])
        Rs1 = sp.linop.Reshape(oshape=A.ishape, ishape=(Nslices, xres, yres))
        Rs2 = sp.linop.Reshape(oshape=(Nslices, Ncoils, xres, yres), ishape=A.oshape)

        SENSE = Rs2 * A * Rs1
        SENSEH = SENSE.H

        # AhA = Ah * A
        # max_eigen = sp.app.MaxEig(AhA, dtype=smaps_sl.dtype, device=spdevice, max_iter=30).run()

        SENSE_torch = sp.to_pytorch_function(SENSE, input_iscomplex=True, output_iscomplex=True)
        SENSEH_torch = sp.to_pytorch_function(SENSEH, input_iscomplex=True, output_iscomplex=True)

        # Initial guess
        imEst = 0.0 * sp.to_pytorch(SENSEH * kspaceU, requires_grad=False)
        #imEst = 0 * SENSEH_torch.apply(kspaceU)

        kspaceU = sp.to_pytorch(kspaceU)

        # forward
        for inner_iter in range(INNER_ITER):
            imEst2 = ReconModel(imEst, kspaceU, SENSE_torch, SENSEH_torch, ifLEARNED=LEARNED)

            if WHICH_LOSS == 'mse':
                loss = mseloss_fcn(imEst2, im)
            elif WHICH_LOSS == 'perceptual':
                loss = loss_perceptual(imEst2, im)
            elif WHICH_LOSS == 'patchGAN':
                loss = loss_GAN(imEst2, im, patchGAN)
            else:
                loss = 0
                for sl in range(imEst2.shape[0]):
                    loss += learnedloss_fcn(imEst2[sl], im[sl], score)
                    torch.cuda.empty_cache()
                loss /= imEst2.shape[0]  # slices average loss
                # undo cropping, imEst2 is square while ReconModel input imEst need to be rectangular
                imEst2 = zero_pad_imEst(imEst2)

            imEst = imEst2

        loss /= INNER_ITER
        print(f'{WHICH_LOSS} loss of batch {i} at inner_iter{inner_iter}, epoch{epoch} is {loss} ')

        eval_avg.update(loss.item(), n=BATCH_SIZE)

        if i == 1:
            # Torch image estimate
            truthplt = torch.abs(torch.squeeze(chan2_complex(im.cpu())))
            perturbed = sp.to_device(sp.from_pytorch(SENSEH_torch.apply(kspaceU)), sp.cpu_device)
            perturbed = chan2_complex(perturbed)
            noisyplt = np.abs(np.squeeze(perturbed))
            noisyplt = torch.from_numpy(noisyplt)

            temp = imEst
            temp = temp.detach().cpu()
            imEstplt = torch.abs(torch.squeeze(chan2_complex(temp)))
            writer_train.add_image('training', imEstplt[2], 0, dataformats='HW')

            with h5py.File(out_name, 'a') as hf:
                if epoch == 0:
                    hf.create_dataset(f"{epoch}_truth", data=torch.unsqueeze(truthplt[2], 0).numpy())
                    hf.create_dataset(f"{epoch}_FT", data=torch.unsqueeze(noisyplt[2], 0).numpy())
                hf.create_dataset(f"{epoch}_recon", data=torch.unsqueeze(imEstplt[2],0).numpy())

            break

        #del smaps, im, kspaceU, imEst2
        #mempool.free_all_blocks()
        #pinned_mempool.free_all_blocks()
        #torch.cuda.empty_cache()

        smaps, im, kspaceU = prefetcher.next()

    print(f'epoch {epoch} took {(time.time() - tt)/60} min')
    #torch.cuda.empty_cache()
    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)

    logging.info('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
    lossT[epoch] = train_avg.avg()
    lossV[epoch] = eval_avg.avg()


