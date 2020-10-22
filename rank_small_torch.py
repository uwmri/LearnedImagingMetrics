import tkinter as tk
from tkinter import filedialog
import h5py as h5
import csv
import logging

from torchvision.models import mobilenet_v2
import torchsummary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch

try:
    from ax.service.managed_loop import optimize
except:
    print('NO ax')


from utils.model_helper import *
from utils.utils_DL import *

train_on_mag = True
shuffle_observers = True
MOBILE = False
EFF = False
BO = False
RESNET = False
ResumeTrain = False
CLIP = False


trainScoreandMSE = True    # train score based classifier and mse(im1)-mse(im2) based classifier at the same time

# Ranks
names = []
root = tk.Tk()
root.withdraw()
filepath_csv = tk.filedialog.askdirectory(title='Choose where the csv file is')
os.chdir(filepath_csv)

files_csv = os.listdir(filepath_csv)
for file in files_csv:
    if fnmatch.fnmatch(file, '*consensus.csv'):
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

# # Human consistency on duplicated pairs
# _, countR = np.unique(ranks, axis=0, return_counts=True)
# _, count = np.unique(ranks[:,2], axis=0, return_counts=True)
# print(f'{ranks.shape[0]-len(count)} Duplicated pairs, {ranks.shape[0]} total pairs')
# print(f'For duplicated pairs, {(ranks.shape[0]-len(countR))/(ranks.shape[0]-len(count))*100} % have the same ranking')

# Shuffle the ranks while the data size is small
if shuffle_observers:
    np.random.shuffle(ranks)

# Examples
maxMatSize = 396  # largest matrix size seems to be 396
NEXAMPLES = 2920
NRANKS = ranks.shape[0]

if train_on_mag:
    nch = 1
else:
    nch = 2

# Images and truth
X_1 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize, nch), dtype=np.float32)  # saved as complex128 though
X_2 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize, nch), dtype=np.float32)
X_T = np.zeros((NEXAMPLES, maxMatSize, maxMatSize, nch), dtype=np.float32)

# X_T = np.zeros((NEXAMPLES, maxMatSize, maxMatSize),dtype=np.complex64)
Labels = np.zeros(NRANKS, dtype=np.int32)

root = tk.Tk()
root.withdraw()
filepath_images = tk.filedialog.askdirectory(title='Choose where the h5 is')
file ='TRAINING_IMAGES_04032020.h5'
file_images = os.path.join(filepath_images, file)
hf = h5.File(name=file_images, mode='r')

log_dir = tk.filedialog.askdirectory(title='Choose log dir')

# Just read and subtract truth for now, index later
for i in range(NEXAMPLES):

    nameT = 'EXAMPLE_%07d_TRUTH' % (i+1)
    name1 = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 0)
    name2 = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 1)


    im1 = zero_pad2D(np.array(hf[name1]), maxMatSize, maxMatSize)
    im2 = zero_pad2D(np.array(hf[name2]), maxMatSize, maxMatSize)
    truth = zero_pad2D(np.array(hf[nameT]), maxMatSize, maxMatSize)

    if train_on_mag:
        X_1[i] = np.sqrt(np.sum(np.square(im1), axis=-1, keepdims=True))
        X_2[i] = np.sqrt(np.sum(np.square(im2), axis=-1, keepdims=True))
        X_T[i] = np.sqrt(np.sum(np.square(truth), axis=-1, keepdims=True))

    else:
        X_1[i] = im1
        X_2[i] = im2
        X_T[i] = truth

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
X_T = np.transpose(X_T, [0, 3, 1, 2])

print(f'X_1 shape {X_1.shape}')

# MobileNet requires values [0,1] and normalized
if MOBILE:
    x1max = X_1.max(axis=(2, 3))
    x1min = X_1.min(axis=(2, 3))
    x2max = X_2.max(axis=(2, 3))
    x2min = X_2.min(axis=(2, 3))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(X_1.shape[0]):
        for ch in range(X_1.shape[1]):
            X_1[i,ch,...] = (X_1[i,ch,...]-x1min[i,ch])/(x1max[i,ch]-x1min[i,ch])
            X_2[i, ch, ...] = (X_2[i, ch, ...] - x2min[i, ch]) / (x2max[i, ch] - x2min[i, ch])

            X_1[i, ch, ...] = (X_1[i, ch, ...] - mean[ch]) / std[ch]
            X_2[i, ch, ...] = (X_2[i, ch, ...] - mean[ch]) / std[ch]

        if i % 1e2 == 0:
            print(f'Normalizing pairs {i + 1}')

ntrain = int(0.9 * NRANKS)
id = ranks[:,2] - 1
idT = id[:ntrain]
idV = id[ntrain:]

Labels_cnnT = Labels[:ntrain]
Labels_cnnV = Labels[ntrain:]

# Data generator
BATCH_SIZE = 16
trainingset = DataGenerator_rank(X_1, X_2, X_T, Labels_cnnT, idT, augmentation=True, pad_channels=0)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

validationset = DataGenerator_rank(X_1, X_2, X_T, Labels_cnnV, idV, augmentation=False, pad_channels=0)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=False)


# check loader, show a batch
check = iter(loader_T)
checkim1, checkim2, checkimt, checklb = check.next()
checkim1 = checkim1.permute(0, 2, 3, 1)
checkim2 = checkim2.permute(0, 2, 3, 1)
checkimt = checkimt.permute(0, 2, 3, 1)
checklbnp = checklb.numpy()

randnum = np.random.randint(16)
imshow(checkim1[randnum, :, :, :].cpu())
imshow(checkim2[randnum, :, :, :].cpu())
imshow(checkimt[randnum, :, :, :].cpu())
print(f'Label is {checklbnp[randnum]}')
# print(f'Label is {checktrans[randnum]}')
# print(f'Label is {checkcrop[randnum]}')

# Ranknet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if MOBILE:
    ranknet = mobilenet_v2(pretrained=False, num_classes=1) # Less than ResNet18
elif EFF:
    ranknet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1, in_channels=1)
elif RESNET:
    ranknet = ResNet2(BasicBlock, [2,2,2,2], for_denoise=False)  # Less than ResNet18
else:
    ranknet = L2cnn(channels_in=X_1.shape[-3])

# Print summary
torchsummary.summary(ranknet, [(X_1.shape[-3], maxMatSize, maxMatSize)
                              ,(X_1.shape[-3], maxMatSize, maxMatSize)], device="cpu")


# Bayesian
# optimize classification accuracy on the validation set as a function of the learning rate and momentum
def train_evaluate(parameterization):

    net = Classifier(ranknet)
    net = train_mod(net=net, train_loader=loader_T, parameters=parameterization, dtype=torch.float, device=device,
                    trainOnMSE=False)
    return evaluate_mod(
        net=net,
        data_loader=loader_V,
        dtype=torch.float,
        device=device,
        trainOnMSE=False
    )


if ResumeTrain:
    # load RankNet
    root = tk.Tk()
    root.withdraw()
    filepath_rankModel = tk.filedialog.askdirectory(title='Choose where the saved metric model is')
    file_rankModel = os.path.join(filepath_rankModel, "RankClassifier5383.pt")
    classifier = Classifier(ranknet)
    #classifier.rank.register_backward_hook(printgradnorm)
    loss_func = nn.CrossEntropyLoss()

    state = torch.load(file_rankModel)
    classifier.load_state_dict(state['state_dict'], strict=True)
    classifier.cuda();
    if trainScoreandMSE:
        file_rankModelMSE = os.path.join(filepath_rankModel, "RankClassifier15_MSE.pt")   
        classifierMSE = Classifier(ranknet)
        classifierMSE.cuda();

    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    optimizer.load_state_dict(state['optimizer'])

    if trainScoreandMSE:
        file_rankModelMSE = os.path.join(filepath_rankModel, "RankClassifier15_MSE.pt")
        mse_module = MSEmodule()
        classifierMSE = Classifier(mse_module)
        stateMSE = torch.load(file_rankModelMSE)
        classifier.load_state_dict(stateMSE['state_dict'], strict=True)
        optimizerMSE = optim.SGD(classifier.parameters(), lr=0.00097, momentum=0.556)
        optimizerMSE.load_state_dict(state['optimizer'])

else:

    classifier = Classifier(ranknet)
    if trainScoreandMSE:
        mse_module = MSEmodule()              
        classifierMSE = Classifier(mse_module)

    if BO:
        best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
                {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            ],
            evaluation_function=train_evaluate,
            objective_name='accuracy',
        )

        optimizer = optim.SGD(classifier.parameters(), lr=best_parameters['lr'], momentum=best_parameters['momentum'])

        print(best_parameters)
        logging.info(f'{best_parameters}')
    else:

        # NEED to set trainOnMSE in train_evaluate manually for both MSE and score-based,
        # get best paramters and initialize optimizier here manually

        #optimizer = optim.SGD(classifier.parameters(), lr=0.003152130338485237, momentum=0.27102874871343374)
        optimizer = optim.Adam(classifier.parameters(), lr=1e-5)
        logging.info('Adam, lr=1e-5')
        if trainScoreandMSE:
            optimizerMSE = optim.SGD(classifierMSE.parameters(), lr=0.00097, momentum=0.556)
            # optimizerMSE = optim.Adam(classifier.parameters(), lr=0.001)

    #classifier.rank.register_backward_hook(printgradnorm)
    loss_func = nn.CrossEntropyLoss()
    classifier.cuda()
    if trainScoreandMSE:
        classifierMSE.cuda()


# Training

from random import randrange
Ntrial = randrange(10000)

# writer = SummaryWriter(f'runs/rank/trial_{Ntrial}')
writer_train = SummaryWriter(os.path.join(log_dir,f'runs/rank/train_{Ntrial}'))
writer_val = SummaryWriter(os.path.join(log_dir,f'runs/rank/val_{Ntrial}'))

logging.basicConfig(filename=os.path.join(log_dir,f'runs/rank/ranking_{Ntrial}.log'), filemode='w', level=logging.INFO)
logging.info('With L2cnn, abs(score) for self.rank')

Nepoch = 200
lossT = np.zeros(Nepoch)
lossV = np.zeros(Nepoch)
# accT = np.zeros(Nepoch)

# keep track of acc of all batches at last epoch
acc_endT = []
acc_end_mseT = []
diff_scoreT = []
diff_mseT = []

acc_endV = []
acc_end_mseV = []
diff_scoreV = []
diff_mseV = []

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

    if trainScoreandMSE:
        train_avgMSE = RunningAverage()
        eval_avgMSE = RunningAverage()
        train_accMSE = RunningAcc()
        eval_accMSE = RunningAcc()

        classifierMSE.train()

    # training
    classifier.train()

    for i, data in enumerate(loader_T, 0):

        # get the inputs
        im1, im2, imt, labels = data             # im (sl, 3 , 396, 396)
        im1, im2, imt = im1.cuda(), im2.cuda(), imt.cuda()
        labels = labels.to(device, dtype=torch.long)

        # classifier
        delta = classifier(im1, im2, imt)

        # loss
        loss = loss_func(delta, labels)
        train_avg.update(loss.item(), n=BATCH_SIZE)  # here is total loss of all batches

        # acc
        acc = acc_calc(delta, labels, BatchSize=BATCH_SIZE)
        #print(f'Training: acc of minibatch {i} is {acc}')
        train_acc.update(acc, n=1)

        # # every 30 minibatch, show image pairs and predictions
        # if i % 30 == 0:
        #     writer.add_figure()
        #if i % 30 == 0:
        #    print(train_acc.avg())

        # zero the parameter gradients, backward and update
        optimizer.zero_grad()
        loss.backward()

        if CLIP:
            clipping_value = 1e-2  # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), clipping_value)

        optimizer.step()


        # train on MSE
        if trainScoreandMSE:
            deltaMSE = classifierMSE(im1, im2, imt)
            lossMSE = loss_func(deltaMSE, labels)
            train_avgMSE.update(lossMSE.item(), n=BATCH_SIZE)  # here is total loss of all batches

            # acc
            accMSE = acc_calc(deltaMSE, labels)
            train_accMSE.update(accMSE, n=1)


#             if i % 30 == 0:
#                 print(f'Acc trained on mse {train_accMSE.avg()}')


            # zero the parameter gradients, backward and update
            optimizerMSE.zero_grad()
            lossMSE.backward()
            if CLIP:
                clipping_value = 1e-2  # arbitrary value of your choosing
                torch.nn.utils.clip_grad_norm_(classifierMSE.parameters(), clipping_value)

            optimizerMSE.step()

        with torch.no_grad():
            score1 = classifier.rank(im1, imt)
            score1 = torch.abs(score1)
            score2 = classifier.rank(im2, imt)
            score2 = torch.abs(score2)
            scorelistT.append(score1.cpu().numpy())
            scorelistT.append(score2.cpu().numpy())

            mse1 = torch.mean((torch.abs(im1 - imt) ** 2), dim=(1, 2, 3))
            mse2 = torch.mean((torch.abs(im2 - imt) ** 2), dim=(1, 2, 3))

            mselistT.append(mse1.cpu().numpy())
            mselistT.append(mse2.cpu().numpy())

    scorelistT = np.concatenate(scorelistT).ravel()
    mselistT = np.concatenate(mselistT).ravel()
    score_mse_figureT = plt_scoreVsMse(scorelistT, mselistT)
    writer_train.add_figure('Score_vs_mse', score_mse_figureT, epoch)


    # validation

    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(loader_V, 0):

            # get the inputs
            im1, im2, imt, labels = data  # im (sl, 3 , 396, 396)
            im1, im2, imt = im1.cuda(), im2.cuda(), imt.cuda()

            labels = labels.to(device, dtype=torch.long)

            # forward
            delta = classifier(im1, im2, imt)

            # loss
            loss = loss_func(delta, labels)
            eval_avg.update(loss.item(), n=BATCH_SIZE)

            # acc
            acc = acc_calc(delta, labels, BatchSize=BATCH_SIZE)
            # print(f'Val: acc of minibatch {i} is {acc}')
            eval_acc.update(acc, n=1)

            # # accuracy
            # _, predictedV = torch.max(delta.data, 1)
            # total += labels.size(0)
            # correct += (predictedV == labels).sum().item()

            # get scores and mse
            score1 = classifier.rank(im1, imt)
            score1 = torch.abs(score1)
            score2 = classifier.rank(im2, imt)
            score2 = torch.abs(score2)
            scorelistV.append(score1.cpu().numpy())
            scorelistV.append(score2.cpu().numpy())

            mse1 = torch.mean((torch.abs(im1 - imt) ** 2), dim=(1, 2, 3))
            mse2 = torch.mean((torch.abs(im2 - imt) ** 2), dim=(1, 2, 3))

            mselistV.append(mse1.cpu().numpy())
            mselistV.append(mse2.cpu().numpy())

            # mse-based classifier
            if trainScoreandMSE:
                deltaMSE = classifierMSE(im1, im2, imt)
                lossMSE = loss_func(deltaMSE, labels)
                eval_avgMSE.update(lossMSE.item(), n=BATCH_SIZE)

                accMSE = acc_calc(deltaMSE, labels, BatchSize=BATCH_SIZE * 2)
                eval_accMSE.update(accMSE, n=1)

    scorelistV = np.concatenate(scorelistV).ravel()
    mselistV = np.concatenate(mselistV).ravel()
    score_mse_figureV = plt_scoreVsMse(scorelistV, mselistV)
    writer_val.add_figure('Score_vs_mse', score_mse_figureV, epoch)

        # accV[epoch] = 100 * correct / total

    #print('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
    print(f'Epoch = {epoch:03d}, Loss = {eval_avg.avg()}, Loss train = {train_avg.avg()}, Acc = {eval_acc.avg()}, Acc train = {train_acc.avg()}')

    if trainScoreandMSE:
        print(f'MSE: Loss = {eval_avgMSE.avg()}, Loss train = {train_avgMSE.avg()}, Acc = {eval_accMSE.avg()}, Acc train = {train_accMSE.avg()}')

    lossT[epoch] = train_avg.avg()
    lossV[epoch] = eval_avg.avg()


    writer_val.add_scalar('Loss', eval_avg.avg(), epoch)
    writer_train.add_scalar('Loss', train_avg.avg(), epoch)

    writer_val.add_scalar('Acc', eval_acc.avg(), epoch)
    writer_train.add_scalar('Acc', train_acc.avg(), epoch)

    if trainScoreandMSE:
        print(f'MSE: Loss = {eval_avgMSE.avg()}, Loss train = {train_avgMSE.avg()}, Acc = {eval_accMSE.avg()}, Acc train = {train_accMSE.avg()}')

        writer_val.add_scalar('LossMSE', eval_avgMSE.avg(),
                              epoch)
        writer_train.add_scalar('LossMSE', train_avgMSE.avg(),
                                epoch)

        writer_val.add_scalar('AccMSE', eval_accMSE.avg(),
                              epoch)
        writer_train.add_scalar('AccMSE', train_accMSE.avg(),
                                epoch)


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

with h5py.File(score_mse_file, 'w') as hf:
    hf.create_dataset(f'scoreT_epoch{Nepoch}', data=scorelistT)
    hf.create_dataset(f'mseT_epoch{Nepoch}', data=mselistT)
    hf.create_dataset(f'scoreV_epoch{Nepoch}', data=scorelistV)
    hf.create_dataset(f'mseV_epoch{Nepoch}', data=mselistV)

# if trainScoreandMSE:
#     acc_endT = np.array(acc_endT)
#     acc_end_mseT = np.array(acc_end_mseT)
#     diff_mseT = np.array(diff_mseT)
#     diff_scoreT = np.array(diff_scoreT)
#
#     acc_endV = np.array(acc_endV)
#     acc_end_mseV = np.array(acc_end_mseV)
#     diff_mseV = np.array(diff_mseV)
#     diff_scoreV = np.array(diff_scoreV)
#
#     plt.plot(acc_endV, acc_end_mseV, '.')
#     plt.title(f'Validation accuracies of minibatches at epoch = {Nepoch}')
#     plt.show()
#
#     plt.plot(acc_endT, acc_end_mseT, '.')
#     plt.title(f'Training accuracies of minibatches at epoch = {Nepoch}')
#     plt.show()
#
#     plt.plot(diff_mseT, acc_end_mseT, '.')
#     plt.title(f'Training accuracy vs MSE at epoch = {Nepoch}')
#     plt.show()
#
#     plt.plot(diff_mseV, acc_end_mseV, '.')
#     plt.title(f'Validation accuracy vs MSE at epoch = {Nepoch}')
#     plt.show()
#
#     plt.plot(diff_scoreT, acc_endT, '.')
#     plt.title(f'Training accuracy vs score difference at epoch = {Nepoch}')
#     plt.show()
#
#     plt.plot(diff_scoreV, acc_endV, '.')
#     plt.title(f'Validation accuracy vs score difference at epoch = {Nepoch}')
#     plt.show()
#
#

