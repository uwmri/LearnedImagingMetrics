from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch

from pathlib import Path
import os
import csv
import h5py as h5

from utils.model_helper import *
from utils import *
from utils.CreateImagePairs import get_smaps, get_truth
from utils.utils_DL import *
from random import randrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Ntrial = 5888
filepath_rankModel = Path(r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn')
filepath_train = Path("I:/NYUbrain")
filepath_val = Path("I:/NYUbrain")
file_rankModel = os.path.join(filepath_rankModel, f"RankClassifier{Ntrial}.pt")
log_dir = filepath_rankModel

rank_channel =1
ranknet = L2cnn(channels_in=rank_channel, channel_base=8, train_on_mag=False)
classifier = Classifier(ranknet)

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
scoreModel = classifier.rank
# for param in classifier.parameters():
#     param.requires_grad = False
scoreModel.cuda()

mse_module = MSEmodule()
ssim_module = SSIM()


NEXAMPLES = 2920
maxMatSize = 396
nch = 1
names = []
filepath_csv = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
os.chdir(filepath_csv)
files_csv = os.listdir(filepath_csv)
for file in files_csv:
    if fnmatch.fnmatch(file, 'consensus_mode_all.csv'):
        names.append(os.path.join(filepath_csv, file))

# Load the ranks
ranks = []
for fname in names:
    print(fname)
    with open(fname) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            ranks.append(row)
ranks = np.array(ranks, dtype=np.int)
NRANKS = ranks.shape[0]

X_1 = np.zeros((NEXAMPLES, nch, maxMatSize, maxMatSize), dtype=np.complex64)
X_2 = np.zeros((NEXAMPLES, nch, maxMatSize, maxMatSize), dtype=np.complex64)
X_T = np.zeros((NEXAMPLES, nch, maxMatSize, maxMatSize), dtype=np.complex64)
Labels = np.zeros(NRANKS, dtype=np.int32)

filepath_images = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
file ='TRAINING_IMAGES_04032020.h5'
file_images = os.path.join(filepath_images, file)
hf = h5.File(name=file_images, mode='r')

# Just read and subtract truth for now, index later
for i in range(NEXAMPLES):

    nameT = 'EXAMPLE_%07d_TRUTH' % (i+1)
    name1 = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 0)
    name2 = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 1)

    im1 = zero_pad2D(np.array(hf[name1]), maxMatSize, maxMatSize)
    im2 = zero_pad2D(np.array(hf[name2]), maxMatSize, maxMatSize)
    truth = zero_pad2D(np.array(hf[nameT]), maxMatSize, maxMatSize)

    # Convert to torch
    X_1[i, 0] = im1[..., 0] + 1j * im1[..., 1]
    X_2[i, 0] = im2[..., 0] + 1j * im2[..., 1]
    X_T[i, 0] = truth[..., 0] + 1j * truth[..., 1]

    if i % 1e2 == 0:
        print(f'loading example pairs {i + 1}')


# All labels
for i in range(0, NRANKS):

    # Label based on ranks from ranker
    if ranks[i, 0] == 2:
        # Same
        Labels[i] = 1
    elif ranks[i, 0] == 1:
        # X_2 is better
        Labels[i] = 0
    else:
        # X_1 is better
        Labels[i] = 2


BATCH_SIZE = 48
id = ranks[:,2] - 1
dataset = DataGenerator_rank(X_1, X_2, X_T, Labels, id, augmentation=True, pad_channels=0)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)



CHECK_BIAS = False
if not CHECK_BIAS:
    scorelist = []
    mselist = []
    ssimlist = []
else:
    scorelist_truth = []
for i, data in enumerate(loader, 0):
    im1, im2, imt, labels = data  # im (sl, ch , 396, 396)
    im1, im2, imt = im1.cuda(), im2.cuda(), imt.cuda()
    labels = labels.to(device, dtype=torch.long)

    if not CHECK_BIAS:
        score1 = scoreModel(im1, imt)
        score2 = scoreModel(im2, imt)
        scorelist.append(score1.detach().cpu().numpy())
        scorelist.append(score2.detach().cpu().numpy())

        mse1 = mse_module(im1, imt)
        mse2 = mse_module(im2, imt)
        mselist.append(mse1.detach().cpu().numpy())
        mselist.append(mse2.detach().cpu().numpy())

        ssim1 = ssim_module(im1, imt)
        ssim2 = ssim_module(im2, imt)
        ssimlist.append(ssim1.detach().cpu().numpy())
        ssimlist.append(ssim2.detach().cpu().numpy())

    else:
        delta, score1, score2 = classifier(imt, imt, imt)
        scorelist_truth.append(score1.detach().cpu().numpy())

if not CHECK_BIAS:
    scorelist = np.concatenate(scorelist).ravel()
    mselist = np.concatenate(mselist).ravel()
    ssimlist = np.concatenate(ssimlist).ravel()

    score_mse_figureT = plt_scoreVsMse(scorelist, mselist)
    score_ssim_figureT = plt_scoreVsMse(scorelist, 1.0 - ssimlist, yname='1.0 - SSIM')

    score_mse_figureT.savefig(f'score_mse_{Ntrial}.png')
    score_ssim_figureT.savefig(f'score_ssim_{Ntrial}.png')
else:
    scorelist_truth = np.concatenate(scorelist_truth).ravel()
