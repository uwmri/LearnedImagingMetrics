import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import io

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


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
        self.sum += value       # sum N batches acc

    def avg(self):
        return self.sum / self.count    # avg acc over batches


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


def plt_scoreVsMse(scorelist, mselist):
    """

    :param scorelistT: list of numbers on cpu(N minibatches, batchsize)
    :param mselistT: list of numbers on cpu(N minibatches, batchsize)
    :param epoch:
    :return: figure
    """

    figure = plt.figure(figsize=(10,10))
    ax = plt.gca()
    plt.scatter(scorelist, mselist,s=150, alpha=0.3)
    plt.xlim([0, 2*np.median(scorelist)])
    plt.ylim([0, 2*np.median(mselist)])
    plt.xlabel('Score', fontsize=24)
    plt.ylabel('MSE', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.yaxis.get_offset_text().set_fontsize(24)

    return figure


# for Bayesian
def train_mod(
    net: torch.nn.Module,
    train_loader: DataLoader,
    parameters: Dict[str, float],
    dtype: torch.dtype,
    device: torch.device

) -> nn.Module:
    """
    Train CNN on provided data set.

    Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.
    """
    # Initialize network
    net.to(dtype=dtype, device=device)  # pyre-ignore [28]
    net.train()
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=parameters.get("lr", 0.001),
        momentum=parameters.get("momentum", 0.0),
        weight_decay=parameters.get("weight_decay", 0.0),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameters.get("step_size", 30)),
        gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
    )
    num_epochs = parameters.get("num_epochs", 1)

    # Train Network
    for _ in range(num_epochs):
        for data in train_loader:

            # get the inputs
            im1, im2, labels = data
            im1, im2 = im1.cuda(), im2.cuda()
            labels = labels.to(device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(im1, im2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return net


def evaluate_mod(
    net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device) -> float:
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            im1, im2, labels = data
            im1, im2, = im1.cuda(), im2.cuda()
            labels = labels.to(device, dtype=torch.long)

            outputs = net(im1, im2)

            # use Acc as metrics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
