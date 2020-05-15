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



def zero_pad2D(input, oshapex, oshapey):
    # zero pad to 396*396 and stack to (n,res,res, 2)

    # pad x
    padxL = int(np.floor((oshapex - input.shape[0])/2))
    padxR = int(oshapex - input.shape[0] - padxL)

    # pad y
    padyU = int(np.floor((oshapey - input.shape[1])/2))
    padyD = int(oshapey - input.shape[1] - padyU)

    input = np.pad(input, ((padxL, padxR),(padyU, padyD)), 'constant', constant_values=0)

    return np.stack((np.real(input),np.imag(input)), axis=-1)


def zero_pad4D(ksp_raw, Nxmax=396, Nymax=768):
    """ zero-pad kspace to the same size (sl, coil, 768, 396)"""

    pady = int(.5 * (Nymax - ksp_raw.shape[2]))
    padx = int(.5 * (Nxmax - ksp_raw.shape[3]))

    ksp_zp = np.pad(ksp_raw, ((0, 0), (0, 0), (pady, Nymax - ksp_raw.shape[2] - pady),
                                 (padx, Nxmax - ksp_raw.shape[3] - padx)), 'constant', constant_values=0 + 0j)
    return ksp_zp


def imshow(im):
    npim = im.numpy()
    npim = np.squeeze(npim)
    abs = np.sqrt(npim[:,:,0]**2 + npim[:,:,1]**2)
    plt.imshow(abs,cmap='gray')
    plt.show()


class ResNet2(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # hard code for now. For 396*396 images, output of last block should be 25*25*512
        # self.fc = nn.Linear(512 * 25 * 25, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DataGenerator(Dataset):
    def __init__(self, X_1, X_2, Y, ID):
        '''
        :param X_1: X_1_cnnT/V
        :param X_2: X_2_cnnT/V
        :param Y: labels
        :param transform
        '''
        self.X_1 = X_1
        self.X_2 = X_2
        self.Y = Y
        self.ID = ID


    def __len__(self):
        return len(self.ID)

    def __getitem__(self, idx):
        IDnum = self.ID[idx]
        x1 = torch.from_numpy(self.X_1[IDnum,...])
        x2 = torch.from_numpy(self.X_2[IDnum,...])
        # x1 = torch.from_numpy(self.X_1[idx, ...])
        # x2 = torch.from_numpy(self.X_2[idx, ...])
        y = self.Y[idx]
        # if self.transform:

        return x1, x2, y


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier,self).__init__()
        self.rank = ResNet2(BasicBlock, [1,1,1,1])
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


def acc_calc(output, labels, BatchSize=16):
    '''
    Returns accuracy of a mini-batch

    '''

    _, preds = torch.max(output, 1)

    return (preds == labels).sum().item()/BatchSize


class RunningAcc:
    def __init__(self):  # initialization
        self.count = 0
        self.sum = 0

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value

    def avg(self):
        return self.sum / self.count
