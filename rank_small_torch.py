import tkinter as tk
from tkinter import filedialog
import h5py as h5
import csv
import logging

from torchvision.models import mobilenet_v2
import torchsummary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

try:
    from ax.service.managed_loop import optimize
except:
    print('NO ax')


from utils.model_helper import *
from utils.utils_DL import *

subtract_truth = True
shuffle_observers = True
MOBILE = False
EFF = False
BO = False
RESNET = False
ResumeTrain = False



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

X_1 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize,2), dtype=np.float32)  # saved as complex128 though
X_2 = np.zeros((NEXAMPLES, maxMatSize, maxMatSize,2), dtype=np.float32)
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
    name = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 0)
    X_1[i, :, :] = zero_pad2D(np.array(hf[name]), maxMatSize, maxMatSize) - zero_pad2D(np.array(hf[nameT]), maxMatSize, maxMatSize)

    name = 'EXAMPLE_%07d_IMAGE_%04d' % (i+1, 1)
    X_2[i, :, :] = zero_pad2D(np.array(hf[name]), maxMatSize, maxMatSize) - zero_pad2D(np.array(hf[nameT]), maxMatSize, maxMatSize)
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
trainingset = DataGenerator_rank(X_1, X_2, Labels_cnnT, idT, augmentation=True)
loader_T = DataLoader(dataset=trainingset, batch_size=BATCH_SIZE, shuffle=True)

validationset = DataGenerator_rank(X_1, X_2, Labels_cnnV, idV, augmentation=False)
loader_V = DataLoader(dataset=validationset, batch_size=BATCH_SIZE, shuffle=True)


# check loader, show a batch
check = iter(loader_T)
checkim1, checkim2, checklb = check.next()
checkim1 = checkim1.permute(0, 2, 3, 1)
checkim2 = checkim2.permute(0, 2, 3, 1)
checklbnp = checklb.numpy()

randnum = np.random.randint(16)
imshow(checkim1[randnum, :, :, :].cpu())
imshow(checkim2[randnum, :, :, :].cpu())
print(f'Label is {checklbnp[randnum]}')
# print(f'Label is {checktrans[randnum]}')
# print(f'Label is {checkcrop[randnum]}')

# Ranknet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if MOBILE:
    ranknet = mobilenet_v2(pretrained=False, num_classes=1) # Less than ResNet18
elif EFF:
    ranknet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
elif RESNET:
    ranknet = ResNet2(BasicBlock, [2,2,2,2], for_denoise=False)  # Less than ResNet18
else:
    ranknet = L2cnn()

torchsummary.summary(ranknet, (3, maxMatSize, maxMatSize), device="cpu")


# Bayesian
# optimize classification accuracy on the validation set as a function of the learning rate and momentum
def train_evaluate(parameterization):

    net = Classifier()
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
    file_rankModel = os.path.join(filepath_rankModel, "RankClassifier15.pt")
    file_rankModelMSE = os.path.join(filepath_rankModel, "RankClassifier15_MSE.pt")
    classifier = Classifier()
    classifierMSE = Classifier()

    state = torch.load(file_rankModel)
    classifier.load_state_dict(state['state_dict'], strict=True)
    optimizer = optim.SGD(classifier.parameters(), lr=0.05045, momentum=0.0)
    optimizer.load_state_dict(state['optimizer'])

    stateMSE = torch.load(file_rankModelMSE)
    classifier.load_state_dict(stateMSE['state_dict'], strict=True)
    optimizerMSE = optim.SGD(classifier.parameters(), lr=0.00097, momentum=0.556)
    optimizerMSE.load_state_dict(state['optimizer'])

else:

    classifier = Classifier(ranknet)
    classifierMSE = Classifier(ranknet)

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

    else:

        # NEED to set trainOnMSE in train_evaluate manually for both MSE and score-based,
        # get best paramters and initialize optimizier here manually

        # optimizer = optim.SGD(classifier.parameters(), lr=0.00043, momentum=1.0)
        optimizerMSE = optim.SGD(classifierMSE.parameters(), lr=0.00097, momentum=0.556)
        optimizer = optim.Adam(classifier.parameters(), lr=1e-5)
        # optimizerMSE = optim.Adam(classifier.parameters(), lr=0.001)


loss_func = nn.CrossEntropyLoss()

classifier.cuda();
classifierMSE.cuda();

# Training

from random import randrange
Ntrial = randrange(10000)

# writer = SummaryWriter(f'runs/rank/trial_{Ntrial}')
writer_train = SummaryWriter(os.path.join(log_dir,f'runs/rank/train_{Ntrial}'))
writer_val = SummaryWriter(os.path.join(log_dir,f'runs/rank/val_{Ntrial}'))

logging.basicConfig(filename=os.path.join(log_dir,f'runs/rank/ranking_{Ntrial}.log'), filemode='w', level=logging.INFO)
logging.info('With L2cnn, Hardswish for self.rank')

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
    train_avgMSE = RunningAverage()
    eval_avgMSE = RunningAverage()

    # To track accuracy
    train_acc = RunningAcc()
    eval_acc = RunningAcc()
    train_accMSE = RunningAcc()
    eval_accMSE = RunningAcc()

    # training
    classifier.train()
    classifierMSE.train()

    # step = 0
    total_steps = len(trainingset)



    for i, data in enumerate(loader_T, 0):

        # get the inputs
        im1, im2, labels = data             # im (sl, 3 , 396, 396)
        im1, im2 = im1.cuda(), im2.cuda()
        labels = labels.to(device, dtype=torch.long)
        # classifier
        delta = classifier(im1, im2, trainOnMSE=False)

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
        if i % 30 == 0:
            print(train_acc.avg())

        # zero the parameter gradients, backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # train on MSE
        deltaMSE = classifierMSE(im1, im2, trainOnMSE=True)
        lossMSE = loss_func(deltaMSE, labels)
        train_avgMSE.update(lossMSE.item(), n=BATCH_SIZE)  # here is total loss of all batches

        # acc
        accMSE = acc_calc(deltaMSE, labels)
        train_accMSE.update(accMSE, n=1)

        if i % 30 == 0:
            print(f'Acc trained on mse {train_accMSE.avg()}')

        # zero the parameter gradients, backward and update
        optimizerMSE.zero_grad()
        lossMSE.backward()
        optimizerMSE.step()

        if epoch == Nepoch-1:
            acc_endT.append(acc)
            acc_end_mseT.append(accMSE)

            diff = torch.sum((torch.abs(im1 - im2) ** 2), dim=(1, 2, 3)) / (im1.shape[1] * im1.shape[2] * im1.shape[3])
            diff = torch.mean(diff)
            diff_mseT.append(diff.item())
            s1 = classifier.rank(im1).detach()
            s1 *= classifier.relu6(s1 + 3) / 6
            s2 = classifier.rank(im2).detach()
            s2 *= classifier.relu6(s2 + 3) / 6
            diff = torch.mean(s1 - s2)
            diff_scoreT.append(diff.item())

            del diff, s1, s2




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
            delta = classifier(im1, im2, trainOnMSE=False)

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

            # mse-based classifier
            deltaMSE = classifierMSE(im1, im2, trainOnMSE=True)
            lossMSE = loss_func(deltaMSE, labels)
            eval_avgMSE.update(lossMSE.item(), n=BATCH_SIZE)

            accMSE = acc_calc(deltaMSE, labels, BatchSize=BATCH_SIZE * 2)
            eval_accMSE.update(accMSE, n=1)

            # save accuracy, mean MSE and mean (score1-score2) for each minibatches
            if epoch == Nepoch-1:
                acc_endV.append(acc)
                acc_end_mseV.append(accMSE)

                diff = torch.sum((torch.abs(im1-im2)**2),dim=(1,2,3))/(im1.shape[1]*im1.shape[2]*im1.shape[3])
                diff = torch.mean(diff)
                diff_mseV.append(diff.item())
                s1 = classifier.rank(im1).detach()
                s1 *= classifier.relu6(s1+3)/6
                s2 = classifier.rank(im2).detach()
                s2 *= classifier.relu6(s2+3)/6
                diff = torch.mean(s1 - s2)
                diff_scoreV.append(diff.item())

                del diff, s1, s2




        # accV[epoch] = 100 * correct / total

    #print('Epoch = %d : Loss Eval = %f , Loss Train = %f' % (epoch, eval_avg.avg(), train_avg.avg()))
    print(f'Epoch = {epoch}, Loss = {eval_avg.avg()}, Loss train = {train_avg.avg()}, Acc = {eval_acc.avg()}, Acc train = {train_acc.avg()}')
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
    torch.save(state,os.path.join(log_dir,f'RankClassifier{Ntrial}_{epoch}.pt'))

    stateMSE = {
        'state_dict': classifierMSE.state_dict(),
        'optimizer': optimizerMSE.state_dict(),
        'epoch': epoch
    }
    torch.save(stateMSE, os.path.join(log_dir, f'RankClassifier{Ntrial}_{epoch}_MSE.pt'))

acc_endT = np.array(acc_endT)
acc_end_mseT = np.array(acc_end_mseT)
diff_mseT = np.array(diff_mseT)
diff_scoreT = np.array(diff_scoreT)

acc_endV = np.array(acc_endV)
acc_end_mseV = np.array(acc_end_mseV)
diff_mseV = np.array(diff_mseV)
diff_scoreV = np.array(diff_scoreV)

plt.plot(acc_endV, acc_end_mseV, '.')
plt.title(f'Validation accuracies of minibatches at epoch = {Nepoch}')
plt.show()

plt.plot(acc_endT, acc_end_mseT, '.')
plt.title(f'Training accuracies of minibatches at epoch = {Nepoch}')
plt.show()

plt.plot(diff_mseT, acc_end_mseT, '.')
plt.title(f'Training accuracy vs MSE at epoch = {Nepoch}')
plt.show()

plt.plot(diff_mseV, acc_end_mseV, '.')
plt.title(f'Validation accuracy vs MSE at epoch = {Nepoch}')
plt.show()

plt.plot(diff_scoreT, acc_endT, '.')
plt.title(f'Training accuracy vs score difference at epoch = {Nepoch}')
plt.show()

plt.plot(diff_scoreV, acc_endV, '.')
plt.title(f'Validation accuracy vs score difference at epoch = {Nepoch}')
plt.show()



