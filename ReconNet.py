import cupy
import h5py as h5
import pickle
import matplotlib.pyplot as plt
import csv
import logging

import torch
import torchvision
import torch.optim as optim
import torchsummary
from torch.utils.tensorboard import SummaryWriter
from ax.service.managed_loop import optimize

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
    filepath_rankModel = Path('D:\git\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
    filepath_train = Path("D:/NYUbrain/brain_multicoil_train")
    filepath_val = Path("D:/NYUbrain/brain_multicoil_val")
file_rankModel = os.path.join(filepath_rankModel, "RankClassifier16.pt")
os.chdir(filepath_rankModel)

classifier = Classifier()

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
score = classifier.rank
score.cuda()

# Get kspace data. (sl, coil, 2, kx, ky)
# root = tk.Tk()
# root.withdraw()
# filepath_raw = tk.filedialog.askdirectory(title='Choose where the raw file is')


scans_train = 'train_20coil.txt'
scans_val = 'val_20coil.txt'

file_train = 'ksp_truths_smaps_train.h5'
file_val = 'ksp_truths_smaps_val.h5'


# Only choose 20-coil data for now
Ncoils = 20
xres = 768
yres = 396
acc = 4

# fixed sampling mask
mask = mri.poisson((xres, yres), accel=acc, crop_corner=True, return_density=False, dtype='float32')
#mask = np.ones((xres, yres), dtype=np.float32)

# Data generator
BATCH_SIZE = 1
trainingset = DataGeneratorRecon(filepath_train, scans_train, file_train, mask)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)
prefetcherT = DataPrefetcher(loader_T)

validationset = DataGeneratorRecon(filepath_val, scans_val, file_val, mask)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=True)
prefetcherV = DataPrefetcher(loader_V)

# # check a training dataset
# sneakpeek(trainingset)
# sneakpeek(validationset)

imSlShape = (BATCH_SIZE,) + (xres, yres)    # (1, 768, 396)

UNROLL = True

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
    # optimizer = optim.SGD(ReconModel.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(ReconModel.parameters(), lr=1e-2)


# training
Ntrial = 2
writer_train = SummaryWriter(f'runs/recon/train_{Ntrial}')
writer_val = SummaryWriter(f'runs/recon/val_{Ntrial}')

WHICH_LOSS = 'mse'
OneNet = True
if WHICH_LOSS == 'perceptual':
    loss_perceptual = PerceptualLoss_VGG16()
    loss_perceptual.cuda()
elif WHICH_LOSS == 'patchGAN':
    patchGAN = NLayerDiscriminator(input_nc=2)
    patchGAN.cuda()

Nepoch = 30
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)

# save some images during training
out_name = os.path.join('sneakpeek_training_CNNshort_mse_norm-truth-max-ksp.h5')
try:
    os.remove(out_name)
except OSError:
    pass

for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    ReconModel.train()

    smaps, im, kspaceU = prefetcherT.next()
    i = 0

    while kspaceU is not None:
        i += 1
        smaps, im, kspaceU = prefetcherT.next()
        # smaps is torch tensor on cpu [1, slice, coil, 768, 396]
        # im is on cuda, ([16, 396, 396, 2])
        # kspaceU is on cuda ([16, 20, 768, 396, 2])

    # for i, data in enumerate(loader_T, 0):
    #
    #     smaps, im, kspaceU = data        # smaps_sl is tensor [1, slice, coil, 768, 396]
        # im *= 1e4                     # im, tensor, [1, slice, 396, 396, 2]
        # im = torch.squeeze(im)
        # kspaceU = torch.squeeze(kspaceU)
        # im, kspaceU = im.cuda(), kspaceU.cuda() # kspaceU, tensor, [1, slice, coil, 768, 396, 2],


        smaps = chan2_complex(np.squeeze(smaps.numpy()))        # (slice, coil, 768, 396)
        smaps = sp.to_device(smaps, device=spdevice)
        Nslices = smaps.shape[0]

        A = sp.linop.Diag(
            [sp.mri.linop.Sense(smaps[s, :, :, :], coil_batch_size=1, ishape=(1, xres, yres)) for s in range(Nslices)])
        Rs1 = sp.linop.Reshape(oshape=A.ishape, ishape=(Nslices, xres, yres))
        Rs2 = sp.linop.Reshape(oshape=(Nslices, Ncoils, xres, yres), ishape=A.oshape)

        SENSE = Rs2 * A * Rs1
        SENSEH = SENSE.H

        # AhA = Ah * A
        # max_eigen = sp.app.MaxEig(AhA, dtype=smaps_sl.dtype, device=spdevice, max_iter=30).run()

        SENSE_torch = sp.to_pytorch_function(SENSE, input_iscomplex=True, output_iscomplex=True)
        SENSEH_torch = sp.to_pytorch_function(SENSEH, input_iscomplex=True, output_iscomplex=True)


        # forward
        imEst = ReconModel(kspaceU,  SENSE_torch, SENSEH_torch)     # torch.Size([slice, 396, 396, 2])

        # loss
        if WHICH_LOSS == 'mse':
            # if OneNet:
            #     loss = loss_fcn_onenet(perturbed, imEst, im, projector, encoder, ClassifierD(Nslices=imEst.shape[0])
            #                            ,ClassifierD_l(Nslices=imEst.shape[0]), lam1=1, lam2=1, lam3=1, lam4=-1, lam5=-1)
            # else:
            loss = mseloss_fcn(imEst, im)
        elif WHICH_LOSS == 'perceptual':
            loss = loss_perceptual(imEst, im)
        elif WHICH_LOSS == 'patchGAN':
            loss = loss_GAN(imEst, im, patchGAN)
        else:
            loss = learnedloss_fcn(imEst, im, score)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_avg.update(loss.item(), n=BATCH_SIZE)
        print(f'Loss for batch {i} = {loss.item()}')

        if i == 9:
            truthplt = torch.abs(torch.squeeze(chan2_complex(im.cpu())))
            perturbed = SENSEH_torch.apply(kspaceU)
            print(f'noisy shape {perturbed.shape}')
            noisyplt = torch.abs(torch.squeeze(chan2_complex(perturbed.cpu())))

            temp = imEst
            temp = temp.detach().cpu()
            imEstplt = torch.abs(torch.squeeze(chan2_complex(temp)))
            writer_train.add_image('training', imEstplt[2], 0, dataformats='HW')

            plt.subplot(131)
            plt.imshow(truthplt[2], cmap='gray')
            plt.title('Truth')
            plt.subplot(132)
            plt.imshow(noisyplt[2], cmap='gray')
            plt.title('FT')
            plt.subplot(133)
            plt.imshow(imEstplt[2], cmap='gray')
            plt.title(f'Epoch = {epoch}, loss = {loss}')
            plt.show()

            with h5py.File(out_name, 'a') as hf:
                hf.create_dataset(f"{epoch}_truth", data=torch.unsqueeze(truthplt[2], 0).numpy())
                hf.create_dataset(f"{epoch}_recon", data=torch.unsqueeze(imEstplt[2],0).numpy())

            print(f'max truth {truthplt[2].max()}')
            print(f'max imEst {imEstplt[2].max()}')

            break


    # Validation
    # with torch.no_grad():
    ReconModel.eval()

    smaps, im, kspaceU = prefetcherV.next()
    i = 0

    while kspaceU is not None:
        i += 1
        smaps, im, kspaceU = prefetcherV.next()

        smaps = chan2_complex(np.squeeze(smaps.numpy()))  # (slice, coil, 768, 396)
        smaps = sp.to_device(smaps, device=spdevice)
        Nslices = smaps.shape[0]

        A = sp.linop.Diag(
            [sp.mri.linop.Sense(smaps[s, :, :, :], coil_batch_size=1, ishape=(1, xres, yres)) for s in range(Nslices)])
        Rs1 = sp.linop.Reshape(oshape=A.ishape, ishape=(Nslices, xres, yres))
        Rs2 = sp.linop.Reshape(oshape=(Nslices, Ncoils, xres, yres), ishape=A.oshape)

        SENSE = Rs2 * A * Rs1
        SENSEH = SENSE.H

        # AhA = Ah * A
        # max_eigen = sp.app.MaxEig(AhA, dtype=smaps_sl.dtype, device=spdevice, max_iter=30).run()

        SENSE_torch = sp.to_pytorch_function(SENSE, input_iscomplex=True, output_iscomplex=True)
        SENSEH_torch = sp.to_pytorch_function(SENSEH, input_iscomplex=True, output_iscomplex=True)

        # forward
        imEst = ReconModel(kspaceU, SENSE_torch, SENSEH_torch)

        if WHICH_LOSS == 'mse':
            loss = mseloss_fcn(imEst, im)
        elif WHICH_LOSS == 'perceptual':
            loss = loss_perceptual(imEst, im)
        elif WHICH_LOSS == 'patchGAN':
            loss = loss_GAN(imEst, im, patchGAN)
        else:
            loss = learnedloss_fcn(imEst, im, score)
        eval_avg.update(loss.item(), n=BATCH_SIZE)

        if i == 1:
            break


    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)

    print('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
    lossT[epoch] = train_avg.avg()
    lossV[epoch] = eval_avg.avg()


