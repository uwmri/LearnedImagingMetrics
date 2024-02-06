import numpy as np
import sigpy as sp
import cupy as cp
import os
import h5py
import  matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import h5py as h5
import csv
import logging
from pathlib import Path
from scipy.stats import pearsonr
from random import randrange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision
from torchinfo import summary
from efficientnet_pytorch import EfficientNet
from torch.utils.tensorboard import SummaryWriter
import argparse
from IQNet import L2cnn, Classifier
from utils.ISOResNet import ISOResNet2, BasicBlock
from utils.utils_DL import DataGenerator_rank, SSIM, MSEmodule, RunningAcc, RunningAverage, acc_calc
from utils.utils import zero_pad2D,plt_scoreVsMse

Ntrial = randrange(10000)
parser = argparse.ArgumentParser()
parser.add_argument('--pname', type=str, default=f'learned_ranking_{Ntrial}')
parser.add_argument('--file_csv', type=str, default=Path(r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\consensus_mode_all.csv'))
parser.add_argument('--file_images', type=str, default=Path(r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\TRAINING_IMAGES_04032020.h5'))
parser.add_argument('--resume_train', action='store_true', default=False)
args = parser.parse_args()

resume_train = args.resume_train
if resume_train:
    Ntrial_prev=4437

train_on_mag = False
shuffle_observers = False
MOBILE = False
EFF = True
BO = False
RESNET = False
CLIP = False
WeightedLoss = False

trainScoreandMSE = False    # train score based classifier and mse(im1)-mse(im2) based classifier at the same time
trainScoreandSSIM = False

maxMatSize = 396  # largest matrix size seems to be 396
NEXAMPLES = 2920

# make sure to use consensus -> consensus_mode.csv
# Load the ranks
ranks = []
print(args.file_csv)
with open(args.file_csv) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        ranks.append(row)
ranks = np.array(ranks, dtype=np.int32)


# import scipy.stats
# uranks_id = np.unique(ranks[:,2])
# new_ranks = np.zeros((len(uranks_id),3), ranks.dtype)
# for count, i in enumerate(uranks_id):
#     #print(f'Use rank {i}')
#     idx = ranks[:,2] == i
#     vals = ranks[idx]
#     #print(vals)
#     m,c = scipy.stats.mode(vals)
#     new_ranks[count,:] = m
# ranks = new_ranks

NRANKS = ranks.shape[0]

# # Human consistency on duplicated pairs
# _, countR = np.unique(ranks, axis=0, return_counts=True)
# _, count = np.unique(ranks[:,2], axis=0, return_counts=True)
# print(f'{ranks.shape[0]-len(count)} Duplicated pairs, {ranks.shape[0]} total pairs')
# print(f'For duplicated pairs, {(ranks.shape[0]-len(countR))/(ranks.shape[0]-len(count))*100} % have the same ranking')

# Shuffle the ranks while the data size is small
if shuffle_observers:
    np.random.shuffle(ranks)
# np.savetxt("consensus_mode_all.csv", ranks, fmt='%d', delimiter=',')

# Images and truth
X_1 = np.zeros((NEXAMPLES, 1, maxMatSize, maxMatSize), dtype=np.complex64)
X_2 = np.zeros((NEXAMPLES, 1, maxMatSize, maxMatSize), dtype=np.complex64)
X_T = np.zeros((NEXAMPLES, 1, maxMatSize, maxMatSize), dtype=np.complex64)
Labels = np.zeros(NRANKS, dtype=np.int32)

with h5.File(name=args.file_images, mode='r') as hf:
    # Just read and subtract truth for now, index later
    for i in range(NEXAMPLES):

        nameT = 'EXAMPLE_%07d_TRUTH' % (i+1)
        name1 = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 0)
        name2 = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 1)

        im1 = zero_pad2D(np.array(hf[name1]), maxMatSize, maxMatSize)
        im2 = zero_pad2D(np.array(hf[name2]), maxMatSize, maxMatSize)
        truth = zero_pad2D(np.array(hf[nameT]), maxMatSize, maxMatSize)

        # Convert to torch
        X_1[i, 0] = im1[...,0] + 1j*im1[...,1]
        X_2[i, 0] = im2[...,0] + 1j*im2[...,1]
        X_T[i, 0] = truth[...,0] + 1j*truth[...,1]

        if i % 1e2 == 0:
            print(f'loading example pairs {i + 1}')

sub1 = X_1 - X_T
sub2 = X_2 - X_T
plt.imshow(np.abs(sub1[10, 0,...]), cmap='gray');plt.axis('off');plt.savefig('sub1_10.png',bbox_inches='tight', pad_inches=0)
plt.imshow(np.abs(sub2[10, 0,...]), cmap='gray');plt.axis('off');plt.savefig('sub2_10.png',bbox_inches='tight', pad_inches=0)

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

print(f'X_1 shape {X_1.shape}')

log_dir = os.path.dirname(args.file_images)
logging.basicConfig(filename=os.path.join(log_dir,f'runs/rank/ranking_{Ntrial}.log'), filemode='w', level=logging.INFO)
logging.info('With ISOresnet classifier')
logging.info(f'{Ntrial}')

CV = 1
CV_fold = 5
logging.info(f'{CV_fold} fold cross validation {CV}')
ntrain = int(0.8 * NRANKS)
id = ranks[:,2] - 1
idV_L = int(NRANKS*(CV-1)*1/CV_fold)
idV_R = int(NRANKS*(CV)*1/CV_fold)
idT = np.concatenate((id[:idV_L], id[idV_R:]))
idV = id[idV_L:idV_R]
logging.info(f'Train on {len(idT)}, eval on {len(idV)} ')
Labels_cnnT = np.concatenate((Labels[:idV_L], Labels[idV_R:]))
Labels_cnnV = Labels[idV_L:idV_R]

# Data generator
BATCH_SIZE = 12
BATCH_SIZE_EVAL = 1
logging.info(f'batchsize={BATCH_SIZE}')
logging.info(f'batchsize_eval={BATCH_SIZE_EVAL}')

trainingset = DataGenerator_rank(X_1, X_2, X_T, Labels_cnnT, idT, augmentation=True, pad_channels=0)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

validationset = DataGenerator_rank(X_1, X_2, X_T, Labels_cnnV, idV, augmentation=False, pad_channels=0)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE_EVAL, shuffle=False, drop_last=True)

if WeightedLoss:
    weight = get_class_weights(Labels)
else:
    weight = torch.ones(3).cuda()


# Ranknet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if resume_train:
    try:
        rank_file = rf'/raid/DGXUserDataRaid/cxt004/NYUbrain/RankClassifier{Ntrial_prev}.pt'
        rank_fileMSE = rf'/raid/DGXUserDataRaid/cxt004/NYUbrain/RankClassifier{Ntrial_prev}_MSE.pt'
        rank_fileSSIM = rf'/raid/DGXUserDataRaid/cxt004/NYUbrain/RankClassifier{Ntrial_prev}_SSIM.pt'
        ranknet = L2cnn(channels_in=1, channel_base=8, group_depth=5, train_on_mag=train_on_mag)

        classifier = Classifier(ranknet)
        mse_module = MSEmodule()
        classifierMSE = Classifier(mse_module)
        ssim_module = SSIM()
        classifierSSIM = Classifier(ssim_module)

        state = torch.load(rank_file)
        classifier.load_state_dict(state['state_dict'], strict=True)
        stateMSE = torch.load(rank_fileMSE)
        classifierMSE.load_state_dict(stateMSE['state_dict'], strict=True)
        stateSSIM = torch.load(rank_fileSSIM)
        classifierSSIM.load_state_dict(stateSSIM['state_dict'], strict=True)
    except:
        print('no recon model found')
else:
    if MOBILE:
        mobilenet_v3_pretrained = torchvision.models.mobilenet_v3_small(pretrained=True)
        in_features = mobilenet_v3_pretrained.classifier[-1].in_features
        mobilenet_v3_pretrained.classifier[-1] = nn.Linear( in_features, 1)
        conv1x1 = nn.Conv2d(2, 3, kernel_size=1)
        ranknet = nn.Sequential(conv1x1, mobilenet_v3_pretrained)
        summary(ranknet.cuda(), input_size=(BATCH_SIZE, 2, 224, 224))

    elif EFF:
        from IQNet import EfficientNet2chan
        ranknet = EfficientNet2chan()
        for param in ranknet.efficientnet._blocks[-7:].parameters():
            param.requires_grad = False
        summary(ranknet.cuda(), input_size=(BATCH_SIZE, 2, 224, 224))

    elif RESNET:
        ranknet = ISOResNet2(BasicBlock, [2,2,2,2], for_denoise=False)  # Less than ResNet18
    else:
        ranknet = L2cnn(channels_in=1, channel_base=8, group_depth=5, train_on_mag=train_on_mag, subtract_truth=False)
        summary(ranknet.cuda(), [(BATCH_SIZE, X_1.shape[-3], maxMatSize, maxMatSize)
            , (BATCH_SIZE, X_1.shape[-3], maxMatSize, maxMatSize)])

    classifier = Classifier(ranknet)
    if trainScoreandMSE:
        mse_module = MSEmodule()
        classifierMSE = Classifier(mse_module)

    if trainScoreandSSIM:
        ssim_module = SSIM()
        classifierSSIM = Classifier(ssim_module)


#
# torchsummary.summary(classifier.cuda(), [(X_1.shape[-3], maxMatSize, maxMatSize)
#                               ,(X_1.shape[-3], maxMatSize, maxMatSize), (X_1.shape[-3], maxMatSize, maxMatSize)])
learning_rate_classifier = 1e-3
learning_rate_rank = 1e-4
learning_rate_MSE = 1e-3
learning_rate_SSIM = 1e-3

optimizer = optim.Adam([
    {'params': classifier.f.parameters()},
    {'params': classifier.g.parameters()},
    {'params': classifier.rank.parameters(), 'lr': learning_rate_rank}
], lr=learning_rate_classifier, weight_decay=1e-5)

logging.info(f'Adam, lr={learning_rate_rank}')
if trainScoreandMSE:
    optimizerMSE = optim.Adam(classifierMSE.parameters(), lr=learning_rate_MSE)

if trainScoreandSSIM:
    optimizerSSIM= optim.Adam(classifierSSIM.parameters(), lr=learning_rate_SSIM)

if resume_train:
    optimizer.load_state_dict(state['optimizer'])
    if trainScoreandMSE:
        optimizerMSE.load_state_dict(stateMSE['optimizer'])
    if trainScoreandSSIM:
        optimizerSSIM.load_state_dict(stateSSIM['optimizer'])

lmbda = lambda epoch: 0.999
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
if trainScoreandMSE:
    schedulerMSE = torch.optim.lr_scheduler.MultiplicativeLR(optimizerMSE, lr_lambda=lmbda)
if trainScoreandSSIM:
    schedulerSSIM = torch.optim.lr_scheduler.MultiplicativeLR(optimizerSSIM, lr_lambda=lmbda)

loss_func = nn.CrossEntropyLoss(weight=weight)

classifier.cuda()
if trainScoreandMSE:
    classifierMSE.cuda()

if trainScoreandSSIM:
    classifierSSIM.cuda()

logging.info(f'learning rate classifier = {learning_rate_classifier}')
logging.info(f'leaning rate rank = {learning_rate_rank}')
if trainScoreandMSE:
    logging.info(f'leaning rate MSE = {learning_rate_MSE}')

if trainScoreandSSIM:
    logging.info(f'leaning rate MSE = {learning_rate_SSIM}')

# Training
writer_train = SummaryWriter(os.path.join(log_dir,f'runs/rank/train_{Ntrial}'))
writer_val = SummaryWriter(os.path.join(log_dir,f'runs/rank/val_{Ntrial}'))

score_mse_file = os.path.join(f'score_mse_file_{Ntrial}.h5')


Nepoch = 2
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)

# keep track of acc of all batches at last epoch
acc_endT = []
acc_end_mseT = []
acc_end_ssimT = []

diff_scoreT = []
diff_mseT = []
diff_ssimT = []

acc_endV = []
acc_end_mseV = []
acc_end_ssimV = []

diff_scoreV = []
diff_mseV = []
diff_ssimV = []

mixed_training = True

if mixed_training:
    scaler = torch.cuda.amp.GradScaler()

for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    # To track accuracy
    train_acc = RunningAcc()
    eval_acc = RunningAcc()

    # track score and mse of images
    scorelistT = []
    scorelistV = []
    mselistT = []
    mselistV = []
    ssimlistT = []
    ssimlistV = []

    if trainScoreandMSE:
        train_avgMSE = RunningAverage()
        eval_avgMSE = RunningAverage()
        train_accMSE = RunningAcc()
        eval_accMSE = RunningAcc()

        classifierMSE.train()

    if trainScoreandSSIM:
        train_avgSSIM = RunningAverage()
        eval_avgSSIM = RunningAverage()
        train_accSSIM = RunningAcc()
        eval_accSSIM = RunningAcc()

        classifierSSIM.train()

    # training
    classifier.train()
    with torch.autograd.set_detect_anomaly(False):
        for i, data in enumerate(loader_T, 0):

            # zero the parameter gradients, backward and update
            optimizer.zero_grad()

            # get the inputs
            im1, im2, imt, labels = data             # im (sl, ch , 396, 396)
            im1, im2, imt = im1.cuda(), im2.cuda(), imt.cuda()
            labels = labels.to(device, dtype=torch.long)


            if MOBILE or EFF:
                im1 -= imt
                im2 -= imt
                # (batch, 1, 396, 396) to (batch, 2, 396, 396)
                im1 = torch.view_as_real(im1).squeeze().permute((0,-1,1,2))
                im2 = torch.view_as_real(im2).squeeze().permute((0,-1,1,2))
                imt = torch.view_as_real(imt).squeeze().permute((0,-1,1,2))

                import torchvision.transforms as transforms
                # these are the mean and std of concat(X_1-X_T, X_2-X_T)
                normalize = transforms.Normalize(mean=[-0.000452, -0.00013541058], std=[0.049449377, 0.050570913])
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),  # Adjust the size based on the specific EfficientNet variant
                    normalize,
                ])

                im1 = torch.stack([preprocess(image) for image in im1])
                im2 = torch.stack([preprocess(image) for image in im2])
                imt = torch.stack([preprocess(image) for image in imt])

            if mixed_training:
                with torch.cuda.amp.autocast():
                    # classifier and scores for each image
                    delta, score1, score2 = classifier(im1, im2, imt)

                    # Cross entropy
                    loss = loss_func(delta, labels)
            else:
                # classifier and scores for each image
                delta, score1, score2 = classifier(im1, im2, imt)

                # Cross entropy
                loss = loss_func(delta, labels)

            # Track loss
            train_avg.update(loss.item(), n=BATCH_SIZE)  # here is total loss of all batches

            # Track accuracy
            acc = acc_calc(delta, labels, BatchSize=BATCH_SIZE)
            #print(f'Training: acc of minibatch {i} is {acc}')
            train_acc.update(acc, n=1)

            if mixed_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if CLIP:
                    clipping_value = 1e-2  # arbitrary value of your choosing
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), clipping_value)
                optimizer.step()

            # train on MSE
            if trainScoreandMSE:
                optimizerMSE.zero_grad()
                deltaMSE, mse1, mse2  = classifierMSE(im1, im2, imt)
                lossMSE = loss_func(deltaMSE, labels)
                train_avgMSE.update(lossMSE.item(), n=BATCH_SIZE)  # here is total loss of all batches

                # acc
                accMSE = acc_calc(deltaMSE, labels, BatchSize=BATCH_SIZE)
                train_accMSE.update(accMSE, n=1)

                # zero the parameter gradients, backward and update
                lossMSE.backward()
                if CLIP:
                    clipping_value = 1e-2  # arbitrary value of your choosing
                    torch.nn.utils.clip_grad_norm_(classifierMSE.parameters(), clipping_value)

                optimizerMSE.step()

            # train on SSIM
            if trainScoreandSSIM:
                optimizerSSIM.zero_grad()
                deltaSSIM, ssim1, ssim2  = classifierSSIM(im1, im2, imt)
                lossSSIM = loss_func(deltaSSIM, labels)
                train_avgSSIM.update(lossSSIM.item(), n=BATCH_SIZE)  # here is total loss of all batches

                # acc
                accSSIM = acc_calc(deltaSSIM, labels, BatchSize=BATCH_SIZE)
                train_accSSIM.update(accSSIM, n=1)

                lossSSIM.backward()
                if CLIP:
                    clipping_value = 1e-2  # arbitrary value of your choosing
                    torch.nn.utils.clip_grad_norm_(classifierSSIM.parameters(), clipping_value)
                optimizerSSIM.step()

            # get the score for a plot
            scorelistT.append(score1.detach().cpu().numpy())
            scorelistT.append(score2.detach().cpu().numpy())

            if trainScoreandMSE:
                mselistT.append(mse1.detach().cpu().numpy())
                mselistT.append(mse2.detach().cpu().numpy())

            if trainScoreandSSIM:
                ssimlistT.append(ssim1.detach().cpu().numpy())
                ssimlistT.append(ssim2.detach().cpu().numpy())

    scorelistT = np.concatenate(scorelistT).ravel()

    if trainScoreandMSE:
        mselistT = np.concatenate(mselistT).ravel()
        score_mse_figureT = plt_scoreVsMse(scorelistT, mselistT)
        corrT, pT = pearsonr(scorelistT, mselistT)
        writer_train.add_figure('Score_vs_mse', score_mse_figureT, epoch)
        writer_train.add_scalar('PearsonCorr', corrT, epoch)
        writer_train.add_scalar('p-value', pT, epoch)

    if trainScoreandSSIM:
        ssimlistT = np.concatenate(ssimlistT).ravel()
        score_ssim_figureT = plt_scoreVsMse(scorelistT, 1.0 - ssimlistT, yname='1.0 - SSIM')
        corrT, pT = pearsonr(scorelistT, ssimlistT)
        writer_train.add_figure('Score_vs_ssim', score_ssim_figureT, epoch)
        writer_train.add_scalar('PearsonCorr_SSIM', corrT, epoch)
        writer_train.add_scalar('p-value_SSIM', pT, epoch)

    scheduler.step()
    if trainScoreandMSE:
        schedulerMSE.step()
    if trainScoreandSSIM:
        schedulerSSIM.step()

    # validation
    classifier.eval()
    if trainScoreandMSE:
        classifierMSE.eval()
    if trainScoreandSSIM:
        classifierSSIM.eval()

    for i, data in enumerate(loader_V, 0):

        # get the inputs
        im1, im2, imt, labels = data  # im (sl, 3 , 396, 396)
        im1, im2, imt = im1.cuda(), im2.cuda(), imt.cuda()

        labels = labels.to(device, dtype=torch.long)

        if MOBILE or EFF:
            im1 -= imt
            im2 -= imt
            # (batch, 1, 396, 396) to (batch, 2, 396, 396)
            im1 = torch.view_as_real(im1).squeeze().permute((-1, 0,1)).unsqueeze(0)
            im2 = torch.view_as_real(im2).squeeze().permute(( -1, 0,1)).unsqueeze(0)
            imt = torch.view_as_real(imt).squeeze().permute(( -1, 0,1)).unsqueeze(0)

            import torchvision.transforms as transforms

            # these are the mean and std of concat(X_1-X_T, X_2-X_T)
            normalize = transforms.Normalize(mean=[-0.000452, -0.00013541058], std=[0.049449377, 0.050570913])
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),  # Adjust the size based on the specific EfficientNet variant
                normalize,
            ])

            im1 = torch.stack([preprocess(image) for image in im1])
            im2 = torch.stack([preprocess(image) for image in im2])
            imt = torch.stack([preprocess(image) for image in imt])

        # forward
        delta, score1, score2 = classifier(im1, im2, imt)

        # loss
        loss = loss_func(delta, labels)

        eval_avg.update(loss.item(), n=BATCH_SIZE_EVAL)

        # acc
        acc = acc_calc(delta, labels, BatchSize=BATCH_SIZE_EVAL)
        # print(f'Val: acc of minibatch {i} is {acc}')
        eval_acc.update(acc, n=1)

        # mse-based classifier. only train those when not MOBILE or EFF due to the expected input of classifier is different for MOBILE or EFF
        if trainScoreandMSE:
            deltaMSE, mse1, mse2 = classifierMSE(im1, im2, imt)
            lossMSE = loss_func(deltaMSE, labels)
            eval_avgMSE.update(lossMSE.item(), n=BATCH_SIZE_EVAL)

            accMSE = acc_calc(deltaMSE, labels, BatchSize=BATCH_SIZE_EVAL)
            eval_accMSE.update(accMSE, n=1)

            mselistV.append(mse1.detach().cpu().numpy())
            mselistV.append(mse2.detach().cpu().numpy())

        if trainScoreandSSIM:
            deltaSSIM, ssim1, ssim2 = classifierSSIM(im1, im2, imt)
            lossSSIM = loss_func(deltaSSIM, labels)
            eval_avgSSIM.update(lossSSIM.item(), n=BATCH_SIZE_EVAL)

            accSSIM = acc_calc(deltaSSIM, labels, BatchSize=BATCH_SIZE_EVAL)
            eval_accSSIM.update(accSSIM, n=1)

            ssimlistV.append(ssim1.detach().cpu().numpy())
            ssimlistV.append(ssim2.detach().cpu().numpy())

        # get scores and mse
        scorelistV.append(score1.detach().cpu().numpy())
        scorelistV.append(score2.detach().cpu().numpy())

    scorelistV = np.concatenate(scorelistV).ravel()

    if trainScoreandMSE:
        mselistV = np.concatenate(mselistV).ravel()
        score_mse_figureV = plt_scoreVsMse(scorelistV, mselistV)
        writer_val.add_figure('Score_vs_mse', score_mse_figureV, epoch)

        # linear correlation
        corrV, pV = pearsonr(scorelistV, mselistV)
        writer_val.add_scalar('PearsonCorr',corrV, epoch)
        writer_val.add_scalar('p-value', corrV, epoch)

    if trainScoreandSSIM:
        ssimlistV = np.concatenate(ssimlistV).ravel()
        score_ssim_figureV = plt_scoreVsMse(scorelistV, 1.0  - ssimlistV, yname='1.0 - SSIM')
        writer_val.add_figure('Score_vs_ssim', score_ssim_figureV, epoch)

        # linear correlation
        corrV, pV = pearsonr(scorelistV, ssimlistV)
        writer_val.add_scalar('PearsonCorr_SSIM',corrV, epoch)
        writer_val.add_scalar('p-value_SSIM', corrV, epoch)

    if trainScoreandSSIM and trainScoreandMSE:
        mse_ssim_figureV = plt_scoreVsMse(mselistV, 1.0 - ssimlistV, xname='MSE', yname='1.0 - SSIM')
        writer_val.add_figure('MSE_vs_ssim', mse_ssim_figureV, epoch)

        # linear correlation
        corrV, pV = pearsonr(mselistV, ssimlistV)
        writer_val.add_scalar('PearsonCorr_SSIM_MSE',corrV, epoch)

    print(f'Epoch = {epoch:03d}, Loss = {eval_avg.avg()}, Loss train = {train_avg.avg()}, Acc = {eval_acc.avg()}, Acc train = {train_acc.avg()}')

    lossT[epoch] = train_avg.avg()
    lossV[epoch] = eval_avg.avg()

    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)

    writer_val.add_scalar('Acc', eval_acc.avg(), epoch)
    writer_train.add_scalar('Acc', train_acc.avg(), epoch)

    if trainScoreandMSE:
        print(f'MSE: Loss = {eval_avgMSE.avg()}, Loss train = {train_avgMSE.avg()}, Acc = {eval_accMSE.avg()}, Acc train = {train_accMSE.avg()}')

        writer_val.add_scalar('LossMSE', eval_avgMSE.avg(), epoch)
        writer_train.add_scalar('LossMSE', train_avgMSE.avg(), epoch)
        writer_val.add_scalar('AccMSE', eval_accMSE.avg(), epoch)
        writer_train.add_scalar('AccMSE', train_accMSE.avg(), epoch)

    if trainScoreandSSIM:
        print(f'SSIM: Loss = {eval_avgSSIM.avg()}, Loss train = {train_avgSSIM.avg()}, Acc = {eval_accSSIM.avg()}, Acc train = {train_accSSIM.avg()}')

        writer_val.add_scalar('LossSSIM', eval_avgSSIM.avg(), epoch)
        writer_train.add_scalar('LossSSIM', train_avgSSIM.avg(), epoch)
        writer_val.add_scalar('AccSSIM', eval_accSSIM.avg(), epoch)
        writer_train.add_scalar('AccSSIM', train_accSSIM.avg(), epoch)

    # save models
    state = {
        'state_dict': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state,os.path.join(log_dir,f'RankClassifier{Ntrial}.pt'))

    if trainScoreandMSE:
        stateMSE = {
            'state_dict': classifierMSE.state_dict(),
            'optimizer': optimizerMSE.state_dict(),
            'epoch': epoch
        }
        torch.save(stateMSE, os.path.join(log_dir, f'RankClassifier{Ntrial}_MSE.pt'))

    if trainScoreandSSIM:
        stateSSIM = {
            'state_dict': classifierSSIM.state_dict(),
            'optimizer': optimizerSSIM.state_dict(),
            'epoch': epoch
        }
        torch.save(stateSSIM, os.path.join(log_dir, f'RankClassifier{Ntrial}_SSIM.pt'))

    with h5py.File(score_mse_file, 'a') as hf:
        hf.create_dataset(f'scoreT_epoch{epoch}', data=scorelistT)
        hf.create_dataset(f'scoreV_epoch{epoch}', data=scorelistV)
        if trainScoreandMSE:
            hf.create_dataset(f'mseV_epoch{epoch}', data=mselistV)
            hf.create_dataset(f'mseT_epoch{epoch}', data=mselistT)
        if trainScoreandSSIM:
            hf.create_dataset(f'ssimV_epoch{epoch}', data=ssimlistV)
            hf.create_dataset(f'ssimT_epoch{epoch}', data=ssimlistT)



