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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
#from livelossplot import PlotLosses
#from torch.utils.tensorboard import SummaryWriter
import torchsummary


def zero_padding(input, oshapex, oshapey):
    # pad x
    padxL = int(np.floor((oshapex - input.shape[0])/2))
    padxR = int(oshapex - input.shape[0] - padxL)

    # pad y
    padyU = int(np.floor((oshapey - input.shape[1])/2))
    padyD = int(oshapey - input.shape[1] - padyU)

    input = np.pad(input, ((padxL, padxR),(padyU, padyD)), 'constant', constant_values=0)
    return input


def imshow(im):
    npim = im.numpy()
    npim = np.squeeze(npim)
    abs = np.sqrt(npim[:,:,0]**2 + npim[:,:,1]**2)
    plt.imshow(abs,cmap='gray')
    plt.show()


subtract_truth = True
shuffle_observers = True

# Ranks
names = []
root = tk.Tk()
filepath_csv = tk.filedialog.askdirectory(title='Choose where the csv file is')
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

# Shuffle the ranks while the data size is small
if shuffle_observers:
    np.random.shuffle(ranks)

# Examples
maxMatSize = 396    # largest matrix size seems to be 396
NEXAMPLES = ranks.shape[0]
X_1 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize),dtype=np.complex64)   # saved as complex128 though
X_2 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize),dtype=np.complex64)
X_T = np.zeros((NEXAMPLES, maxMatSize, maxMatSize),dtype=np.complex64)
Labels = np.zeros(NEXAMPLES, dtype=np.int32)

root = tk.Tk()
filepath_images = tk.filedialog.askdirectory(title='Choose where the h5 is')
for file in os.listdir(filepath_images):
    if fnmatch.fnmatch(file, '*.h5'):
        file_images = os.path.join(filepath_images, file)
hf = h5.File(name=file_images, mode='r')

# Make a real database of images
for i in range(0, ranks.shape[0]):
    # print('Example %d, Image %d ( %d %d)' % (i, ranks[i, 2], ranks[i, 0], ranks[i, 1]))

    name = 'EXAMPLE_%07d_TRUTH' % (ranks[i,2])
    X_T[i, :, :] = zero_padding(np.array(hf[name]), maxMatSize, maxMatSize)

    name = 'EXAMPLE_%07d_IMAGE_%04d' % (ranks[i, 2], 0)
    X_1[i, :, :] = zero_padding(np.array(hf[name]), maxMatSize, maxMatSize)

    name = 'EXAMPLE_%07d_IMAGE_%04d' % (ranks[i, 2], 1)
    X_2[i, :, :] = zero_padding(np.array(hf[name]), maxMatSize, maxMatSize)

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
print(f'Training set size {X_1_cnnT.shape}')
print(f'Validation set size {X_1_cnnV.shape}')


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

randnum = np.random.randint(16)
imshow(checkim1[randnum,:,:,:])
imshow(checkim2[randnum,:,:,:])
print(f'Label is {checklbnp[randnum]}')



# Ranknet setup
channels_cnn = 16
kernel_size = 3
stride_size = 1
image_size = maxMatSize
padding_size = 1


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels_out)

        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels_out)

        # shortcut, do conv2d with 1*1 kernel to reshape when channels_in != out
        self.shortcut = nn.Sequential()
        if channels_out != channels_in or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=(1,1), stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(channels_out)
            )

    def forward(self, c):
        out = F.relu(self.bn1(self.conv1(c)))
        out = self.bn2(self.conv2(out))
        # print(f'shape of out is {out.shape}')
        out += self.shortcut(c)
        short = self.shortcut(c)
        # print(f'shape of shortcut {short.shape}')
        # print(f'shape of out+shortcut is {out.shape}')
        out = F.relu(out)
        # print(f'shape of final out is {out.shape}')
        return out



class Ranknet(nn.Module):

    def __init__(self):
        super(Ranknet,self).__init__()
        # first conv, fixed out channels = channel_cnn, kernel of 3*3
        self.conv1 = nn.Conv2d(2, channels_cnn, kernel_size=kernel_size,stride=1,padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels_cnn)

        # stages
        self.stage1 = self._stage(16, 16, 3, stride=1)
        self.stage2 = self._stage(16, 32, 3, stride=2)
        self.stage3 = self._stage(32, 64, 3, stride=2)

        # FC
        self.convfull = nn.Conv2d(64,1,kernel_size=(16,16))

    def _stage(self, channels_in, channels_out, kernel, stride):
        return nn.Sequential(
            ResBlock(channels_in, channels_out, kernel, stride=stride),
            ResBlock(channels_out, channels_out, kernel, stride=1)
        )

    def forward(self, c):
        out = F.relu(self.bn1(self.conv1(c)))
        # print(f'shape after initial conv3by3 {out.shape}')
        out = self.stage1(out)
        # print(f'shape after 1st stage {out.shape}')     # [16, 16, 256, 256]
        out = self.stage2(out)
        # print(f'shape after 2nd stage {out.shape}')
        out = self.stage3(out)
        # print(f'shape after 3rd stage {out.shape}')
        out = nn.AvgPool2d(4)(out)
        # print(f'shape after average pool {out.shape}')

        out = self.convfull(out)
        # print(f'shape after pseudo fc {out.shape}')
        return out

    # def num_flat_features(self,c):
    #     size = c.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features
ranknet = Ranknet()
torchsummary.summary(ranknet, (2,maxMatSize, maxMatSize), device="cpu")

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier,self).__init__()
        self.rank = Ranknet()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,3)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, image1,image2):
        score1 = self.rank(image1)
        score2 = self.rank(image2)
        # print(score2.shape)
        score1 = score1.view(score1.shape[0], -1)
        score2 = score2.view(score2.shape[0], -1)
        # print(f'shape of score2 after reshape {score2.shape}')
        d = score1 - score2
        # print(d.shape)
        d = torch.tanh(self.fc1(d))
        d = self.drop(d)
        d = F.softmax(self.fc2(d))
        return d

classifier = Classifier()


optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
classifier.cuda()



# Tensorboard
# writer = SummaryWriter('runs/rank_small')
# get some random training images
# Grabbed above, checkim1, checkim2, checklb

# # create grid of images
# img_grid = torchvision.utils.make_grid(images)
#
# # show images
# matplotlib_imshow(img_grid, one_channel=True)
#
# # write to tensorboard
# writer.add_image('four_fashion_mnist_images', img_grid)

# visualize the model
# writer.add_graph(ranknet, checkim1, checkim2)
# writer.close()

class RunningAverage:
    def __init__(self):  # initialization
        self.count = 0
        self.sum = 0

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n

    def avg(self):
        return self.sum / self.count


# liveloss = PlotLosses()

# Training
for epoch in range(10):

    logs = {}
    # Setup counter to keep track of loss
    train_avg = RunningAverage()
    eval_avg = RunningAverage()


    running_loss = 0.0

    # training
    classifier.train()

    step = 0
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
        train_avg.update(loss.item(), n=BATCH_SIZE)  # here is loss

        if step % 10 == 0:
            print('Step %d of %d : %f' % (step, total_steps, train_avg.avg()))

        # zero the parameter gradients, backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1



        running_loss += loss.item()


    # validation
    classifier.eval()
    step = 0
    for i, data in enumerate(loader_V, 0):
        # get the inputs
        im1, im2, labels = data
        im1, im2, = im1.cuda(), im2.cuda()
        labels = labels.to(device, dtype=torch.long)

        # forward
        delta = classifier(im1, im2)

        # loss
        loss = loss_func(delta, labels)
        eval_avg.update(loss.item(), n=BATCH_SIZE)

        print(f'validation loss is {loss}')

    print(f'training average loss is {train_avg.avg()}')
    print(f'training average loss is {eval_avg.avg()}')
    # logs['loss'] = train_avg.avg()
    # logs['val_loss'] = eval_avg.avg()

    # liveloss.update(logs)
    # liveloss.draw()

    print('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))






