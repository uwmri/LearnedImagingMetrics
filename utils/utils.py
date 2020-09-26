import h5py
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

def imshow(im):
    npim = im.numpy()
    npim = np.squeeze(npim)
    abs = np.sqrt(npim[:,:,0]**2 + npim[:,:,1]**2)
    plt.imshow(abs,cmap='gray')
    plt.show()


def get_slices(ksp_zp, Nsl):
    """ get N slices from each case, return fully sampled (and undersampled ksp) """
    idx = np.random.randint(0, ksp_zp.shape[0], Nsl)

    return ksp_zp[idx, ...]


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
