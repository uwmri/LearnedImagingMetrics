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
    filepath_rankModel = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
    filepath_train = Path("D:/NYUbrain/brain_multicoil_train")
    filepath_val = Path("D:/NYUbrain/brain_multicoil_val")

    # filepath_rankModel = Path('E:\ImagePairs_Pack_04032020')
    # filepath_train = Path("E:/")
    # filepath_val = Path("E:/")

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


# Only choose 20-coil data for now
Ncoils = 20
xres = 768
yres = 396
acc = 8

# fixed sampling mask
WHICH_MASK='none'
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
trainingset = DataGeneratorRecon(filepath_train, scans_train, file_train, mask)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


validationset = DataGeneratorRecon(filepath_val, scans_val, file_val, mask)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


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
    return evaluate_modRecon(
        net=net,
        data_loader=loader_V,
        dtype=torch.float,
        device=device
    )


# training
Ntrial = 2.52
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

INNER_ITER = 1

Nepoch = 50
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)

logging.basicConfig(filename=f'Recon_{Ntrial}.log', filemode='w', level=logging.INFO)

# save some images during training
out_name = os.path.join(f'sneakpeek_training{Ntrial}.h5')
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
    # optimizer = optim.SGD(ReconModel.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(ReconModel.parameters(), lr=LR)

    logging.info(f'Adam, lr = {LR}')

logging.info('case averaged loss')
logging.info('Denoiser = CNN_short, with 2 blocks')
logging.info(f'inner_iter = {INNER_ITER}')
logging.info('9 cases training, 1 cases validation')
logging.info(f'loss = {WHICH_LOSS}')
logging.info(f'acc = {acc}, mask is {WHICH_MASK}')

ifSingleSlice = True
logging.info(f'ifSingleSlice={ifSingleSlice}')

for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    ReconModel.train()
    tt = time.time()

    prefetcher = DataPrefetcher(loader_T)

    smaps, im, kspaceU = prefetcher.next()
    i = 0

    while kspaceU is not None:
        t = time.time()
        i += 1

        # smaps is torch tensor on cpu [1, slice, coil, 768, 396]
        # im is on cuda, ([16, 396, 396, 2])
        # kspaceU is on cuda ([16, 20, 768, 396, 2])

        smaps = sp.to_device(smaps, device=spdevice)
        #smaps = spdevice.xp.squeeze(smaps)
        smaps = chan2_complex(spdevice.xp.squeeze(smaps))  # (slice, coil, 768, 396)
        Nslices = smaps.shape[0]
        #print(f'Load batch {time.time()-t}, {Nslices} {smaps.shape}')

        t = time.time()
        if ifSingleSlice:
            loss_temp = 0
            for sl in range(Nslices):
                t_sl = time.time()
                smaps_sl = smaps[sl]                                # ndarray on cuda (20, 768, 396), complex64
                im_sl = im[sl]                                      # tensor on cuda (768, 396, 2)
                kspaceU_sl = kspaceU[sl]                            # tensor on cuda (20, 768, 396, 2)
                with spdevice:
                    A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
                    Ah = A.H
                    # A ishape (768, 396), oshape (20, 768, 396)
                    # Ah ishape (20,768,396), oshape(768,396)

                A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
                Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)

                imEst = 0 * Ah_torch.apply(kspaceU_sl)      # imEst shape (768, 396, 2)

                t = time.time()
                optimizer.zero_grad()
                for inner_iter in range(INNER_ITER):
                    imEst2 = ReconModel(kspaceU_sl, A_torch, Ah_torch, imEst)    # (768, 396, 2)
                    if WHICH_LOSS == 'mse':
                        loss_temp = mseloss_fcn(imEst2, im_sl)
                    elif WHICH_LOSS == 'perceptual':
                        loss_temp = loss_perceptual(imEst2, im_sl)
                    elif WHICH_LOSS == 'patchGAN':
                        loss_temp = loss_GAN(imEst2, im_sl, patchGAN)
                    else:
                        loss_temp = learnedloss_fcn(imEst2, im_sl, score)
                    loss = loss_temp
                    loss.backward(retain_graph=True)
                    imEst = imEst2

                print(f'case {i}, slice {sl}, took {time.time() - t_sl}')

                optimizer.step()
            #     loss_temp += loss_temp
            # loss_temp /= Nslices
            # loss = loss_temp
            # loss.backward(retain_graph=True)
            #
            # optimizer.step()



        else:
            A = sp.linop.Diag(
                [sp.mri.linop.Sense(smaps[s, :, :, :], coil_batch_size=None) for s in range(Nslices)])
            Rs1 = sp.linop.Reshape(oshape=A.ishape, ishape=(Nslices, xres, yres))
            Rs2 = sp.linop.Reshape(oshape=(Nslices, Ncoils, xres, yres), ishape=A.oshape)

            SENSE = Rs2 * A * Rs1
            SENSEH = SENSE.H

            # AhA = Ah * A
            # max_eigen = sp.app.MaxEig(AhA, dtype=smaps_sl.dtype, device=spdevice, max_iter=30).run()

            SENSE_torch = sp.to_pytorch_function(SENSE, input_iscomplex=True, output_iscomplex=True)
            SENSEH_torch = sp.to_pytorch_function(SENSEH, input_iscomplex=True, output_iscomplex=True)

            imEst = 0 * SENSEH_torch.apply(kspaceU)     # torch.Size([16, 768, 396, 2])

            t = time.time()
            optimizer.zero_grad()
            for inner_iter in range(INNER_ITER):
                imEst2 = ReconModel(kspaceU, SENSE_torch, SENSEH_torch, imEst)
                if WHICH_LOSS == 'mse':
                    loss = mseloss_fcn(imEst2, im)
                    print(f'mse loss of batch {i} at inner_iter{inner_iter}, epoch{epoch} is {loss} ')
                elif WHICH_LOSS == 'perceptual':
                    loss = loss_perceptual(imEst2, im)
                elif WHICH_LOSS == 'patchGAN':
                    loss = loss_GAN(imEst2, im, patchGAN)
                else:
                    loss = learnedloss_fcn(imEst2, im, score)
                    torch.cuda.empty_cache()

                loss.backward(retain_graph=True)
                imEst = imEst2

                print(f'Train: learned loss of batch {i}, epoch{epoch} is {loss} ')
            optimizer.step()

            if epoch%10 == 0:
                logging.info(f'Inner iteration took {time.time()-t}s')

        train_avg.update(loss.item(), n=BATCH_SIZE)
        logging.info(f'Training Loss for batch {i} = {loss.item()}')

        if i == 9:
            break

        del smaps,im, kspaceU, imEst2
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        torch.cuda.empty_cache()
        # print(f'Total time = {time.time() - tt}')
        #exit()

        smaps, im, kspaceU = prefetcher.next()

    # Validation
    # with torch.no_grad():
    ReconModel.eval()
    prefetcher = DataPrefetcher(loader_V)

    smaps, im, kspaceU = prefetcher.next()
    i = 0
    while kspaceU is not None:
        i += 1

        #smaps = chan2_complex(np.squeeze(smaps.numpy()))  # (slice, coil, 768, 396)
        smaps = sp.to_device(smaps, device=spdevice)
        smaps = chan2_complex(spdevice.xp.squeeze(smaps))
        Nslices = smaps.shape[0]

        if ifSingleSlice:
            loss_temp = 0
            for sl in range(smaps.shape[0]):
                smaps_sl = smaps[sl]
                im_sl = im[sl]
                kspaceU_sl = kspaceU[sl]

                with spdevice:
                    A = sp.mri.linop.Sense(smaps_sl, coil_batch_size=None, weights=mask_gpu)
                    Ah = A.H

                A_torch = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
                Ah_torch = sp.to_pytorch_function(Ah, input_iscomplex=True, output_iscomplex=True)

                imEst = 0 * Ah_torch.apply(kspaceU_sl)      # (768, 396, 2)

                t = time.time()
                optimizer.zero_grad()
                for inner_iter in range(INNER_ITER):
                    imEst2 = ReconModel(kspaceU_sl, A_torch, Ah_torch, imEst)
                    if WHICH_LOSS == 'mse':
                        loss_temp = mseloss_fcn(imEst2, im_sl)
                    elif WHICH_LOSS == 'perceptual':
                        loss_temp = loss_perceptual(imEst2, im_sl)
                    elif WHICH_LOSS == 'patchGAN':
                        loss_temp = loss_GAN(imEst2, im_sl, patchGAN)
                    else:
                        loss_temp = learnedloss_fcn(imEst2, im_sl, score)
                    # loss_temp += loss_temp
                    loss = loss_temp
                    imEst = imEst2



                eval_avg.update(loss.item(), n=BATCH_SIZE)

                if i == 1 and sl == 2:
                    truthplt = torch.abs(chan2_complex(im_sl.cpu()))
                    perturbed = Ah_torch.apply(kspaceU_sl)
                    noisyplt = torch.abs(chan2_complex(perturbed.cpu()))

                    temp = imEst
                    temp = temp.detach().cpu()
                    imEstplt = torch.abs(chan2_complex(temp))

                    # truthplt = torch.unsqueeze(truthplt, 0)
                    # noisyplt = torch.unsqueeze(noisyplt, 0)
                    # imEstplt = torch.unsqueeze(imEstplt, 0)

                    with h5py.File(out_name, 'a') as hf:
                        if epoch == 0:
                            hf.create_dataset(f"{epoch}_truth", data=truthplt.numpy())
                            hf.create_dataset(f"{epoch}_FT", data=noisyplt.numpy())
                            hf.create_dataset(f"{epoch}_recon", data=imEstplt.numpy())
                        else:
                            hf.create_dataset(f"{epoch}_recon", data=imEstplt.numpy())




            # loss_temp /= Nslices
            # loss = loss_temp
            # eval_avg.update(loss.item(), n=BATCH_SIZE)

            if i == 1:
                break


        else:
            A = sp.linop.Diag(
                [sp.mri.linop.Sense(smaps[s, :, :, :], coil_batch_size=None) for s in range(Nslices)])
            Rs1 = sp.linop.Reshape(oshape=A.ishape, ishape=(Nslices, xres, yres))
            Rs2 = sp.linop.Reshape(oshape=(Nslices, Ncoils, xres, yres), ishape=A.oshape)

            SENSE = Rs2 * A * Rs1
            SENSEH = SENSE.H

            # AhA = Ah * A
            # max_eigen = sp.app.MaxEig(AhA, dtype=smaps_sl.dtype, device=spdevice, max_iter=30).run()

            SENSE_torch = sp.to_pytorch_function(SENSE, input_iscomplex=True, output_iscomplex=True)
            SENSEH_torch = sp.to_pytorch_function(SENSEH, input_iscomplex=True, output_iscomplex=True)

            # Initial guess
            imEst = 0 * SENSEH_torch.apply(kspaceU)

            # forward
            for inner_iter in range(INNER_ITER):
                imEst2 = ReconModel(kspaceU, SENSE_torch, SENSEH_torch, imEst)

                if WHICH_LOSS == 'mse':
                    loss = mseloss_fcn(imEst2, im)
                elif WHICH_LOSS == 'perceptual':
                    loss = loss_perceptual(imEst2, im)
                elif WHICH_LOSS == 'patchGAN':
                    loss = loss_GAN(imEst2, im, patchGAN)
                else:
                    loss = learnedloss_fcn(imEst2, im, score)
                    torch.cuda.empty_cache()

                imEst = imEst2


            print(f'Val: learned loss of batch {i}, epoch{epoch} is {loss} ')

                # loss = loss / 20.0


            eval_avg.update(loss.item(), n=BATCH_SIZE)

            if i == 1:
                truthplt = torch.abs(torch.squeeze(chan2_complex(im.cpu())))
                perturbed = SENSEH_torch.apply(kspaceU)
                noisyplt = torch.abs(torch.squeeze(chan2_complex(perturbed.cpu())))

                temp = imEst
                temp = temp.detach().cpu()
                imEstplt = torch.abs(torch.squeeze(chan2_complex(temp)))
                writer_train.add_image('training', imEstplt[2], 0, dataformats='HW')

                with h5py.File(out_name, 'a') as hf:
                    if epoch == 0:
                        hf.create_dataset(f"{epoch}_truth", data=torch.unsqueeze(truthplt[2], 0).numpy())
                        hf.create_dataset(f"{epoch}_FT", data=torch.unsqueeze(noisyplt[2], 0).numpy())
                        hf.create_dataset(f"{epoch}_recon", data=torch.unsqueeze(imEstplt[2],0).numpy())
                    else:
                        hf.create_dataset(f"{epoch}_recon", data=torch.unsqueeze(imEstplt[2], 0).numpy())

                break

        del smaps, im, kspaceU, imEst2
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        torch.cuda.empty_cache()

        smaps, im, kspaceU = prefetcher.next()

    print(f'epoch {epoch} took {(time.time() - tt)/60} min')
    torch.cuda.empty_cache()
    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)

    logging.info('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
    lossT[epoch] = train_avg.avg()
    lossV[epoch] = eval_avg.avg()


