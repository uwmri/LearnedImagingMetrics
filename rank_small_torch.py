from typing import Any

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import csv

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from livelossplot import PlotLosses

subtract_truth = True
shuffle_observers = True

# Ranks
names = ('ranks_CT_const.csv', 'ranks_CT_const2.csv')
# Load the ranks
ranks = []
for fname in names:
    print(fname)
    with open(fname) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            ranks.append(row)
ranks = np.array(ranks, dtype=np.int)

# Shuffle the ranks while the data size is small
if shuffle_observers:
    np.random.shuffle(ranks)

# Examples
NEXAMPLES = ranks.shape[0]
X_1 = np.zeros((NEXAMPLES,256,256),dtype=np.complex64)
X_2 = np.zeros((NEXAMPLES,256,256),dtype=np.complex64)
X_T = np.zeros((NEXAMPLES,256,256),dtype=np.complex64)
Labels = np.zeros(NEXAMPLES, dtype=np.int32)

hf = h5.File(name='TRAINING_IMAGES_V1.h5', mode='r')

# Make a real database of images
for i in range(0, ranks.shape[0]):
    # print('Example %d, Image %d ( %d %d)' % (i, ranks[i, 2], ranks[i, 0], ranks[i, 1]))

    name = 'EXAMPLE_%06d_TRUTH' % (ranks[i,2])
    X_T[i, :, :] = np.array(hf[name])

    name = 'EXAMPLE_%06d_IMAGE_%04d' % (ranks[i, 2], 0)
    X_1[i, :, :] = np.array(hf[name])

    name = 'EXAMPLE_%06d_IMAGE_%04d' % (ranks[i, 2], 1)
    X_2[i, :, :] = np.array(hf[name])

    # Label based on ranks from ranker
    if ranks[i,0] == 2:
        # Same
        Labels[i] = 1
    elif ranks[i,0] == 1:
        # X_2 is better
        Labels[i] = 0
    else:
        # X_1 is better
        Labels[i] = 2


if subtract_truth == True:
    X_1 -= X_T
    X_2 -= X_T

X_1_cnn = np.stack((np.real(X_1), np.imag(X_1)), axis=-1)
X_2_cnn = np.stack((np.real(X_2), np.imag(X_2)), axis=-1)
print(X_1_cnn.shape)

ntotal = NEXAMPLES
ntrain = int(0.9*NEXAMPLES)
X_1_cnnT = X_1_cnn[:ntrain,:,:,:]
X_2_cnnT = X_2_cnn[:ntrain,:,:,:]
Labels_cnnT = Labels[:ntrain]
X_1_cnnV = X_1_cnn[ntrain:,:,:,:]
X_2_cnnV = X_2_cnn[ntrain:,:,:,:]
Labels_cnnV = Labels[ntrain:]

# torch tensor should be minibatch * channel * H*W
X_1_cnnT = np.transpose(X_1_cnnT,[0,3,1,2])
X_2_cnnT = np.transpose(X_2_cnnT,[0,3,1,2])
X_1_cnnV = np.transpose(X_1_cnnV,[0,3,1,2])
X_2_cnnV = np.transpose(X_2_cnnV,[0,3,1,2])
print(X_1_cnnT.shape)

# Transform


# Data generator
class DataGenerator(Dataset):
    def __init__(self, X_1, X_2, Y):
        '''
        :param X_1: X_1_cnnT/V
        :param X_2: X_2_cnnT/V
        :param Y: labels
        :param transform
        '''
        self.X_1 = X_1
        self.X_2 = X_2
        self.Y = Y
        self.len = X_1.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x1 = torch.from_numpy(self.X_1[idx,...])
        x2 = torch.from_numpy(self.X_2[idx,...])
        y = self.Y[idx]
        # if self.transform:

        return x1, x2, y

BATCH_SIZE = 16
trainingset = DataGenerator(X_1_cnnT,X_2_cnnT,Labels_cnnT)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

validationset = DataGenerator(X_1_cnnV,X_2_cnnV,Labels_cnnV)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=True)

# check loader, show a batch
check = iter(loader_T)
checkim1,checkim2,checklb = check.next()
checkim1 = checkim1.permute(0,2,3,1)
checkim2 = checkim2.permute(0,2,3,1)
checklbnp = checklb.numpy()

def imshow(im):
    npim = im.numpy()
    npim = np.squeeze(npim)
    abs = np.sqrt(npim[:,:,0]**2 + npim[:,:,1]**2)
    plt.imshow(abs,cmap='gray')
    plt.show()

randnum = np.random.randint(16)
imshow(checkim1[randnum,:,:,:])
imshow(checkim2[randnum,:,:,:])
print(f'Label is {checklbnp[randnum]}')



# Ranknet setup
channels_cnn = 16
kernel_size = 5
stride_size = 2
image_size = 256
padding_size = np.int(np.floor((image_size*(stride_size-1)-stride_size+kernel_size)/2))
class Ranknet(nn.Module):

    def __init__(self, channels, kernel, stride, padding):
        super(Ranknet,self).__init__()
        self.conv1 = nn.Conv2d(2, channels,kernel,stride=stride, padding=[padding_size,padding_size], bias=True)
        self.conv2 = nn.Conv2d(channels_cnn, 2*channels_cnn, kernel_size, stride=stride, padding=[padding_size,padding_size], bias=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(231200, 1)

        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, c1, c2):
        c1 = F.max_pool2d(F.relu(self.conv1(c1)),3)
        c1 = self.drop(c1)
        c1 = F.avg_pool2d(F.leaky_relu(self.conv2(c1),negative_slope=0.03),2)
        c1 = self.drop(c1)
        c1 = c1.view(-1, self.num_flat_features(c1))   # flatten
        c1 = self.fc(c1)

        c2 = F.max_pool2d(F.relu(self.conv1(c2)), 3)
        c2 = self.drop(c2)
        c2 = F.avg_pool2d(F.leaky_relu(self.conv2(c2), negative_slope=0.03), 2)
        c2 = self.drop(c2)
        c2 = c2.view(-1, self.num_flat_features(c2))  # flatten
        c2 = self.fc(c2)

        diff = c1 - c2
        diff = F.tanh(self.fc1(diff))
        diff = self.drop(diff)
        diff = F.softmax(self.fc2(diff))

        return diff

    def num_flat_features(self,c):
        size = c.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


ranknet = Ranknet(channels_cnn,kernel_size,stride_size,padding_size)
print(ranknet)

for param in ranknet.parameters():
    print(type(param.data), param.size())

# class Classifier(nn.Module):
#
#     def __init__(self):
#         super(Classifier,self).__init__()
#         self.fc1 = nn.Linear(1,8)
#         self.fc2 = nn.Linear(8,3)
#         self.drop = nn.Dropout(p=0.5)
#
#     def forward(self, d):
#         d = F.tanh(self.fc1(d))
#         d = self.drop(d)
#         d = F.softmax(self.fc2(d))
#         return d
#
# classifier = Classifier()
# print(classifier)


optimizer = optim.SGD(ranknet.parameters(), lr=0.001, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
ranknet.cuda()

# liveloss = PlotLosses()

# Training
for epoch in range(10):

    loss_realtime = 0.0

    # training
    ranknet.train()

    for i, data in enumerate(loader_T, 0):
        # get the inputs
        im1, im2, labels = data
        im1, im2 = im1.cuda(), im2.cuda()
        labels = labels.to(device, dtype=torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        delta = ranknet(im1, im2)
        loss = loss_func(delta, labels)
        loss.backward()
        optimizer.step()



    # validation
    ranknet.eval()
    for i, data in enumerate(loader_V, 0):
        # get the inputs
        im1, im2, labels = data
        im1, im2, labels = im1.cuda(), im2.cuda(), labels.cuda()

        delta = ranknet(im1, im2)
        loss = loss_func(delta, labels)

        print(f'validation loss is {loss}')








