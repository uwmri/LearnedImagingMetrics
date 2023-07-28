import h5py
import cupy
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
import fnmatch
import os
import torch
import sigpy as sp
import sigpy.mri as mri
import sigpy.plot as pl


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                if os.path.islink(name) == False:
                    result.append(os.path.join(root, name))
    return result


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

def zero_pad3D(ksp_raw, Nxmax=396, Nymax=768):

    pady = int(.5 * (Nymax - ksp_raw.shape[-2]))
    padx = int(.5 * (Nxmax - ksp_raw.shape[-1]))

    ksp_zp = np.pad(ksp_raw, ((0, 0), (pady, Nymax - ksp_raw.shape[-2] - pady),
                                 (padx, Nxmax - ksp_raw.shape[-1] - padx)), 'constant', constant_values=0 + 0j)
    return ksp_zp



def zero_pad_truth(truth, Nymax=768):
    ''' zero pad truth from ([16, 396, 396, 2]) to ([16, 768, 396, 2]) '''
    pady = int(.5 * (Nymax - truth.shape[1]))
    truth_zp = np.pad(truth, ((0, 0), (pady, Nymax - truth.shape[1] - pady), (0,0), (0,0)), 'constant', constant_values=0)

    return truth_zp

def zero_pad_imEst(image, Nymax=768):
    '''input: image torch tensor (sl, 396, 396, 2)'''
    pady = int(.5 * (Nymax - image.shape[1]))
    padyD = int(Nymax - image.shape[1] - pady)
    image = torch.nn.functional.pad(image, (0,0,0,0,pady, padyD), mode='constant', value=0)
    return image


def crop_im(image):
    '''crop imEst and truth to torch([sl, 396,396,2])'''

    idxL = int((image.shape[1] - image.shape[2]) / 2)
    idxR = int(idxL + image.shape[2])
    image = image[:, idxL:idxR, ...]
    return image




def complex_2chan(input):
    # input is cuda array, complex64
    # output is cuda array, float32, (,2)

    if torch.is_tensor(input):
        output = torch.zeros(input.shape + (2,), dtype=torch.float32)
        output[..., 0] = torch.real(input)
        output[..., 1] = torch.imag(input)
        return output

    xp = sp.get_device(input).xp
    #input = sp.to_device(input, sp.cpu_device)

    output = xp.zeros(input.shape+(2,), dtype=np.float32)
    output[..., 0] = xp.real(input)
    output[..., 1] = xp.imag(input)

    return output


def chan2_complex(input):
    output = input[...,0] + 1j *input[...,1]

    return output


def print_mem():
    cupy_mempool = cupy.get_default_memory_pool()
    cupy_pinned_mempool = cupy.get_default_pinned_memory_pool()
    torch_allocated = torch.cuda.memory_allocated()
    torch_max_allocated = torch.cuda.max_memory_allocated()
    torch_cached = torch.cuda.memory_reserved()
    print(f'cupy mem {cupy_mempool.used_bytes()*9.313e-10} Gb')
    print(f'torch mempool {torch_allocated*9.313e-10} Gb')
    print(f'torch max mempool {torch_max_allocated*9.313e-10} Gb')
    print(f'torch reserved {torch_cached*9.313e-10} Gb')


def pad_beginning(data, prob):
    # data: dataframe
    # prob: 1d numpy array
    # fix bug when dataframe doen't start with [ID,1,Prob, Reviewer]
    if data.iloc[0]['Results'] == 10:
        prob = np.insert(prob,0, 0, axis=0)
    elif data.iloc[0]['Results'] == 22:
        prob = np.insert(prob,0, (0,0), axis=0)
    else:
        pass
    return prob


def add_zero_prob(reviewerdata, num_intersection):

    reviewer_results = reviewerdata['Results'].to_numpy()
    reviewer_ID = reviewerdata['ID'].to_numpy()

    idx_incomp = []
    jj = np.arange(0, len(reviewer_results), 1)
    j = 0
    idx_incomp.append(j)
    for i in range(1, len(reviewer_results)):
        if reviewer_ID[i] == reviewer_ID[i - 1]:
            if reviewer_results[i - 1] == 1:
                if reviewer_results[i] == 10:
                    j += 1
                else:
                    j += 2
            else:
                j += 1
        else:
            if reviewer_results[i - 1] == 1:
                if reviewer_results[i] == 1:
                    j += 3
                elif reviewer_results[i] == 10:
                    j += 4
                else:
                    j += 5
            elif reviewer_results[i - 1] == 10:
                if reviewer_results[i] == 1:
                    j += 2
                elif reviewer_results[i] == 10:
                    j += 3
                else:
                    j += 4
            else:
                if reviewer_results[i] == 1:
                    j += 1
                elif reviewer_results[i] == 10:
                    j += 2
                else:
                    j += 3

        idx_incomp.append(j)
    idx_incomp = np.array(idx_incomp)

    reviewerdata['j'] = idx_incomp
    reviewerdata = reviewerdata.set_index('j')

    # this will fill missing data as nan
    reviewerdata = reviewerdata.reindex(index=np.arange(num_intersection * 3))
    reviewerdata['Prob'].fillna(0, inplace=True)

    return reviewerdata


def plt_scoreVsMse(scorelist, mselist, xname='Learned Score', yname='MSE', add_regression=False):
    """

    :param scorelistT: 1d np array on cpu(N minibatches*batchsize, )
    :param mselistT: 1d np array on cpu(N minibatches*batchsize, )
    :param epoch:
    :return: figure
    """

    figure = plt.figure(figsize=(10,10))
    ax = plt.gca()
    plt.scatter(scorelist, mselist,s=150, alpha=0.3)

    # Add regression
    if add_regression:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(scorelist, mselist)
        x = np.linspace( 0, np.max(scorelist),100)
        line = slope * x + intercept
        plt.plot(x, line, 'k', label='y={:.2f}x+{:.2f}'.format(slope, intercept), linewidth=5)


    # plt.xlim([0, 2*np.median(scorelist)])
    # plt.ylim([0, 2*np.median(mselist)])
    plt.xlabel(xname, fontsize=24)
    plt.ylabel(yname, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.yaxis.get_offset_text().set_fontsize(24)

    return figure


def plt_loss_learnedVsMse(loss_learned, loss_mse, xname='Learned Loss', yname='MSE'):

    figure = plt.figure(figsize=(10,10))
    ax = plt.gca()
    plt.scatter(loss_learned, loss_mse,s=150, alpha=0.3)
    plt.xlim([0, 2*np.median(loss_learned)])
    plt.ylim([0, 2*np.median(loss_mse)])
    plt.xlabel(xname, fontsize=24)
    plt.ylabel(yname, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.yaxis.get_offset_text().set_fontsize(24)

    return figure


def plt_recon(recon):

    figure = plt.figure(figsize=(10,10))
    plt.imshow(recon.numpy(), cmap='gray')

    return figure


def plt_scores(score1, score2):
    if not isinstance(score1,np.ndarray):
        score1 = score1.detach().cpu().numpy()
        score2 = score2.detach().cpu().numpy()
    figure = plt.figure(figsize=(10, 10))
    plt.scatter(score1.squeeze(), score2.squeeze())
    # plt.xlabel('score of unshifted/unscaled')
    # plt.ylabel('score of shifted/scaled')
    plt.xlabel('score of (imt1, imt2))')
    plt.ylabel('score of (imt, im1))')

    return figure

