import tkinter as tk
from tkinter import filedialog
import fnmatch
import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import csv

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchsummary

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from model_helper import *

subtract_truth = True
shuffle_observers = True

# Ranks
names = []
root = tk.Tk()
root.withdraw()
filepath_csv = tk.filedialog.askdirectory(title='Choose where the csv file is')
os.chdir(filepath_csv)

files_csv = os.listdir(filepath_csv)
for file in files_csv:
    if fnmatch.fnmatch(file, '*.csv'):
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

# Human consistency on duplicated pairs
_, countR = np.unique(ranks, axis=0, return_counts=True)
_, count = np.unique(ranks[:,2], axis=0, return_counts=True)
print(f'{ranks.shape[0]-len(count)} Duplicated pairs, {ranks.shape[0]} total pairs')
print(f'For duplicated pairs, {(ranks.shape[0]-len(countR))/(ranks.shape[0]-len(count))*100} % have the same ranking')

# Shuffle the ranks while the data size is small
if shuffle_observers:
    np.random.shuffle(ranks)

# Examples
maxMatSize = 396  # largest matrix size seems to be 396
NEXAMPLES = 2920
NRANKS = ranks.shape[0]

X_1 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize,2), dtype=np.float32)  # saved as complex128 though
X_2 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize,2), dtype=np.float32)
# X_T = np.zeros((NEXAMPLES, maxMatSize, maxMatSize),dtype=np.complex64)
Labels = np.zeros(NRANKS, dtype=np.int32)

root = tk.Tk()
root.withdraw()
filepath_images = tk.filedialog.askdirectory(title='Choose where the h5 is')
for file in os.listdir(filepath_images):
    if fnmatch.fnmatch(file, '*.h5'):
        file_images = os.path.join(filepath_images, file)
hf = h5.File(name=file_images, mode='r')

# Just read and subtract truth for now, index later
for i in range(NEXAMPLES):

    nameT = 'EXAMPLE_%07d_TRUTH' % (i+1)
    name = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 0)
    X_1[i, :, :] = zero_pad2D(np.array(hf[name]), maxMatSize, maxMatSize) - zero_pad2D(np.array(hf[nameT]), maxMatSize,
                                                                                       maxMatSize)

    name = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 1)
    X_2[i, :, :] = zero_pad2D(np.array(hf[name]), maxMatSize, maxMatSize) - zero_pad2D(np.array(hf[nameT]), maxMatSize,
                                                                                       maxMatSize)
    if i % 1e2 == 0:
        print(f'loading example pairs {i + 1}')

# All labels
for i in range(0, ranks.shape[0]):

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

# torch tensor should be minibatch * channel * H*W
X_1 = np.transpose(X_1, [0, 3, 1, 2])
X_2 = np.transpose(X_2, [0, 3, 1, 2])
print(X_1.shape)

ntrain = int(0.9 * NRANKS)
id = ranks[:,2] - 1
idT = id[:ntrain]
idV = id[ntrain:]

Labels_cnnT = Labels[:ntrain]
Labels_cnnV = Labels[ntrain:]

# X_1_cnnT = X_1[idx[:ntrain],...]
# X_2_cnnT = X_2[idx[:ntrain], ...]
# X_1_cnnV = X_1[idx[ntrain:], ...]
# X_2_cnnV = X_2[idx[ntrain:], ...]
#
# # torch tensor should be minibatch * channel * H*W
# X_1_cnnT = np.transpose(X_1_cnnT, [0, 3, 1, 2])
# X_2_cnnT = np.transpose(X_2_cnnT, [0, 3, 1, 2])
# X_1_cnnV = np.transpose(X_1_cnnV, [0, 3, 1, 2])
# X_2_cnnV = np.transpose(X_2_cnnV, [0, 3, 1, 2])
# print(f'Training set size {X_1_cnnT.shape}')
# print(f'Validation set size {X_1_cnnV.shape}')

# Data generator
BATCH_SIZE = 16
# trainingset = DataGenerator(X_1_cnnT, X_2_cnnT, Labels_cnnT)
trainingset = DataGenerator(X_1, X_2, Labels_cnnT, idT)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

# validationset = DataGenerator(X_1_cnnV, X_2_cnnV, Labels_cnnV)
validationset = DataGenerator(X_1, X_2, Labels_cnnV, idV)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE * 2, shuffle=True)

# check loader, show a batch
check = iter(loader_T)
checkim1, checkim2, checklb = check.next()
checkim1 = checkim1.permute(0, 2, 3, 1)
checkim2 = checkim2.permute(0, 2, 3, 1)
checklbnp = checklb.numpy()

randnum = np.random.randint(16)
imshow(checkim1[randnum, :, :, :])
imshow(checkim2[randnum, :, :, :])
print(f'Label is {checklbnp[randnum]}')

# Ranknet
ranknet = ResNet2(BasicBlock, [1,1,1,1])  # Less than ResNet18
# torchsummary.summary(ranknet, (2, maxMatSize, maxMatSize), device="cpu")

classifier = Classifier()
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
classifier.cuda();

# Training
Ntrial = 2
# writer = SummaryWriter(f'runs/rank/trial_{Ntrial}')
writer_train = SummaryWriter(f'runs/rank/train_{Ntrial}')
writer_val = SummaryWriter(f'runs/rank/val_{Ntrial}')

Nepoch = 100
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)
accV = np.zeros(Nepoch)
for epoch in range(Nepoch):

    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()

    # To track accuracy
    train_acc = RunningAcc()
    eval_acc = RunningAcc()

    # running_loss = 0.0

    # training
    classifier.train()

    # step = 0
    total_steps = len(trainingset)

    for i, data in enumerate(loader_T, 0):

        # get the inputs
        im1, im2, labels = data
        im1, im2 = im1.cuda(), im2.cuda()
        labels = labels.to(device, dtype=torch.long)



        # forward
        delta = classifier(im1, im2)

        # loss
        loss = loss_func(delta, labels)
        train_avg.update(loss.item(), n=BATCH_SIZE)  # here is total loss of all batches

        # acc
        acc = acc_calc(delta, labels)
        # print(f'Training: acc of minibatch {i} is {acc}')
        train_acc.update(acc, n=1)

        # # every 30 minibatch, show image pairs and predictions
        # if i % 30 == 0:
        #     writer.add_figure()

        # zero the parameter gradients, backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # writer.add_scalar('Loss/Train',
    #                   train_avg.avg(),
    #                   epoch)
    # writer.add_scalar('Accuracy/Train',
    #                   train_acc.avg(),
    #                   epoch)

    # validation
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(loader_V, 0):

            # get the inputs
            im1, im2, labels = data

            # print(f'Val: im1 shape {im1.shape}')
            # print(f'Val: label shape {labels.shape}')

            im1, im2, = im1.cuda(), im2.cuda()
            labels = labels.to(device, dtype=torch.long)

            # forward
            delta = classifier(im1, im2)

            # loss
            loss = loss_func(delta, labels)
            eval_avg.update(loss.item(), n=BATCH_SIZE)

            # acc
            acc = acc_calc(delta, labels, BatchSize=BATCH_SIZE*2)
            # print(f'Val: acc of minibatch {i} is {acc}')
            eval_acc.update(acc, n=1)

            # accuracy
            _, predictedV = torch.max(delta.data, 1)
            total += labels.size(0)
            correct += (predictedV == labels).sum().item()

        accV[epoch] = 100 * correct / total

    print('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
    lossT[epoch] = train_avg.avg()
    lossV[epoch] = eval_avg.avg()

    # writer.add_scalar('Loss',
    #                   eval_avg.avg(),
    #                   epoch)
    # writer.add_scalar('Accuracy/Val',
    #                   eval_acc.avg(),
    #                   epoch)

    # Display train and val on the same graph
    writer_val.add_scalar('Loss', eval_avg.avg(),
                          epoch)
    writer_train.add_scalar('Loss', train_avg.avg(),
                            epoch)

    writer_val.add_scalar('Acc', eval_acc.avg(),
                          epoch)
    writer_train.add_scalar('Acc', train_acc.avg(),
                            epoch)

state = {
    'state_dict': classifier.state_dict(),
    'optimizer': optimizer.state_dict()
}
torch.save(state, 'RankClassifier2.pt')

# Save

# torch.save(ranknet, 'RankScoring.pt')

# Plot classifier
# copied from keras version
# test_input = np.linspace(start=-10,stop=10,num=1000)
# test_input = np.reshape(test_input, (1000,1))
# test_output = classifier_model.predict(test_input)
#
# plt.figure()
# plt.plot(test_input,test_output)
# plt.show()
