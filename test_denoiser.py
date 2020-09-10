#
import sigpy as sp
import sigpy.mri as mri
from torch.utils.tensorboard import SummaryWriter
from ax.service.managed_loop import optimize

from utils.Recon_helper import *
from utils.utils_DL import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load RankNet
filepath_rankModel = Path('D:\git\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
file_rankModel = os.path.join(filepath_rankModel, "RankClassifier16.pt")
os.chdir(filepath_rankModel)

classifier = Classifier()

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
score = classifier.rank
score.cuda()

# load truth and noisy images
filepath_train = Path("D:/NYUbrain/brain_multicoil_train")
filepath_val = Path("D:/NYUbrain/brain_multicoil_val")

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
# mask = mri.poisson((xres, yres), accel=acc, crop_corner=True, return_density=False, dtype='float32')
mask = np.ones((xres, yres), dtype=np.float32)

# Data generator
BATCH_SIZE = 1
trainingset = DataGeneratorDenoise(filepath_train, scans_train, file_train)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

validationset = DataGeneratorDenoise(filepath_val, scans_val, file_val)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=True)

denoiser = Unet()
denoiser.cuda()
BO = False

# Bayesian
# optimize on MSE of validation image vs. truth (no noise added)
def train_evaluate(parameterization):

    net = DnCNN()
    net = train_mod(net=net, train_loader=loader_T, parameters=parameterization, dtype=torch.float, device=device,
                    trainOnMSE=False)
    return evaluate_mod(
        net=net,
        data_loader=loader_V,
        dtype=torch.float,
        device=device,
        trainOnMSE=True
    )
if BO:
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-3, 1e2], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        ],
        evaluation_function=train_evaluate,
        objective_name='accuracy',
    )

    optimizer = optim.SGD(classifier.parameters(), lr=best_parameters['lr'], momentum=best_parameters['momentum'])

    print(best_parameters)

else:
    optimizer = optim.Adam(denoiser.parameters(), lr=0.001)
    #optimizer = optim.SGD(classifier.parameters(), lr=1, momentum=0.9)


# training
Ntrial = 2
writer_train = SummaryWriter(f'runs/denoise/train_{Ntrial}')
writer_val = SummaryWriter(f'runs/denoise/val_{Ntrial}')

WHICH_LOSS = 'mse'
if WHICH_LOSS == 'perceptual':
    loss_perceptual = PerceptualLoss_VGG16()
    loss_perceptual.cuda()
elif WHICH_LOSS == 'patchGAN':
    patchGAN = NLayerDiscriminator(input_nc=2)
    patchGAN.cuda()

Nepoch = 100
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)

for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    denoiser.train()

    for i, data in enumerate(loader_T, 0):
        im, im_noisy = data
        im *= 1e4
        im_noisy  *= 1e4
        im, im_noisy = torch.squeeze(im), torch.squeeze(im_noisy)
        im, im_noisy = im.cuda(), im_noisy.cuda()

        im_denoise = denoiser(im_noisy)

        # loss
        if WHICH_LOSS == 'mse':
            loss = mseloss_fcn(im, im_denoise)
        elif WHICH_LOSS == 'perceptual':
            loss = loss_perceptual(im, im_denoise)
        elif WHICH_LOSS == 'patchGAN':
            loss = loss_GAN(im, im_denoise, patchGAN)
        else:
            loss = learnedloss_fcn(im, im_denoise, score)

        train_avg.update(loss.item(), n=BATCH_SIZE)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i==1:
            if epoch%3==0:
                inputplt = im_noisy.cpu()
                inputplt = torch.abs(chan2_complex(inputplt.permute(0,2,3,1)))

                temp = im_denoise
                temp = temp.detach().cpu()
                imEstplt = torch.abs(chan2_complex(temp.permute(0, 2, 3, 1)))

                plt.subplot(121)
                plt.imshow(inputplt[2], cmap='gray')
                plt.title('Input image')
                plt.subplot(122)
                plt.imshow(imEstplt[2], cmap='gray')
                plt.title(f'Epoch = {epoch}, loss = {loss}')
                plt.show()
        if i==18:
            break

    denoiser.eval()
    for i, data in enumerate(loader_V, 0):
        im, im_noisy = data
        im *= 1e4
        im_noisy *= 1e4
        im, im_noisy = torch.squeeze(im), torch.squeeze(im_noisy)
        im, im_noisy = im.cuda(), im_noisy.cuda()

        im_denoise = denoiser(im_noisy)

        # loss
        if WHICH_LOSS == 'mse':
            loss = mseloss_fcn(im, im_denoise)
        elif WHICH_LOSS == 'perceptual':
            loss = loss_perceptual(im, im_denoise)
        elif WHICH_LOSS == 'patchGAN':
            loss = loss_GAN(im, im_denoise, patchGAN)
        else:
            loss = learnedloss_fcn(im, im_denoise, score)

        eval_avg.update(loss.item(), n=BATCH_SIZE)

        if i == 2:
            break

    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)

    print('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
    lossT[epoch] = train_avg.avg()
    lossV[epoch] = eval_avg.avg()



