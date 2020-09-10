import h5py
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

import sigpy as sp
import sigpy.mri as mri
import fnmatch
import os

from utils.CreateImagePairs import trans_motion


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                if os.path.islink(name) == False:
                    result.append(os.path.join(root, name))
    return result

def get_slice_nums(tot_slices, num_slices):

    # Get list of slice numbers without duplicates
    # tot_slices: total number of slices of the scan
    # num_slices: number of slices you want to show from each case

    mu, sigma = 0, 0.7 * tot_slices
    for ii in range(10000):  # Not ideal...
        num_duplicates = 0
        num_outrange = 0
        index_duplicates = []
        slice_nums = np.floor(np.abs(np.random.normal(mu, sigma, num_slices)))
        for i in range(0, num_slices):
            if slice_nums[i] >= tot_slices:
                num_outrange += 1
            for j in range(i + 1, num_slices):
                if slice_nums[i] == slice_nums[j]:
                    index_duplicates.append(j)
                    num_duplicates += 1
        if num_duplicates == 0 and num_outrange == 0:
            break
    return slice_nums


def get_smaps(kspacegpu=None):
    smaps = sp.mri.app.EspiritCalib(kspacegpu, calib_width=24, thresh=0.005, kernel_width=7, crop=0.8,
                                    max_iter=50,
                                    device=device, show_pbar=True).run()
    return smaps


def get_truth(kspacegpu=None, mps=None, lamda=None):
      # (coil, h, w)

    image_sense = mri.app.SenseRecon(kspacegpu, mps, lamda=lamda, device=device,
                                     max_iter=20).run()
    # crop and flip up down
    width = image_sense.shape[1]
    if width < 320:
        image_sense = image_sense[160:480, :]
    else:
        image_sense = image_sense[.5 * width:1.5 * width, :]

    # send to cpu and normalize
    image_sense = sp.to_device(image_sense, sp.cpu_device)
    image_sense = np.flipud(image_sense)
    image_sense = np.abs(image_sense) / np.max(np.abs(image_sense))

    return image_sense


def get_l1tv(kspacegpu=None, mps=None, lamda=None, num_iter=None):
    # L1TV and crop and flip
    print(f'l1TV recon with lambda={lamda}')
    image = mri.app.TotalVariationRecon(kspacegpu, mps, lamda=lamda, device=device,
                                        max_iter=num_iter).run()
    image = sp.to_device(image, sp.cpu_device)

    # crop and flip up down

    width = image.shape[1]
    image_sq = []
    if width < 320:
        image_sq = image[160:480, :]
    else:
        lower = int(.5 * width)
        upper = int(1.5 * width)
        image_sq = image[lower:upper, :]

    image_sq = np.flipud(image_sq)
    image_sq = np.abs(image_sq)/np.max(np.abs(image_sq))
    return image_sq

def get_l1wavelet(kspacegpu=None, mps=None, lamda=None, num_iter=None):
    # L1TV and crop and flip
    print(f'l1TV recon with lambda={lamda}')
    image = mri.app.L1WaveletRecon(kspacegpu, mps, lamda=lamda, device=device,
                                        max_iter=num_iter).run()
    image = sp.to_device(image, sp.cpu_device)

    # crop and flip up down

    width = image.shape[1]
    image_sq = []
    if width < 320:
        image_sq = image[160:480, :]
    else:
        lower = int(.5 * width)
        upper = int(1.5 * width)
        image_sq = image[lower:upper, :]

    image_sq = np.flipud(image_sq)
    image_sq = np.abs(image_sq)/np.max(np.abs(image_sq))
    return image_sq



def mse(x, y):
    return np.linalg.norm(x - y)


train_folder_5 = Path("D:/NYUbrain/brain_multicoil_train_5")
files = find("*.h5", train_folder_5)  # list of file paths

device = sp.Device(0)

for index_file, file in enumerate(files):

    # get rid of files with less than 8 coils
    file_size = os.path.getsize(file)
    if file_size < 300000000:
        break

    hf = h5py.File(file)

    ksp = hf['kspace'][()]
    ksp_width = ksp.shape[3]
    ksp_height = ksp.shape[2]

    xv, yv = np.meshgrid(np.linspace(-1, 1, ksp_height), np.linspace(-1, 1, ksp_width), sparse=False, indexing='ij')
    radius = np.sqrt(np.square(xv) + np.square(yv))
    print(f'radius shape is {radius.shape}')

    slice_nums = get_slice_nums(np.size(ksp, 0), 1)

    diff_tot = []
    for s in slice_nums:
        s = int(s)
        ksp_gpu = sp.to_device(ksp[s], device=device)
        smaps = get_smaps(ksp_gpu)
        diff = []
        for recon in range(0, 2):

            # TRUTH by espirit
            if recon == 0:
                image_sense = get_truth(kspacegpu=ksp_gpu, mps=smaps, lamda=0.005)

            else:
                acc_base = np.random.randint(0, 20)

                # Create a sampling mask
                mask = np.random.random_sample(ksp[s, 0, :, :].shape)
                variable_density = np.random.rand()
                sampling_density = np.ones(ksp[s, 0, :, :].shape) * variable_density + (1 - variable_density) / (
                            0.01 + radius)
                mask *= sampling_density

                acc = acc_base + np.random.randint(0, 10)

                thresh = np.percentile(mask, acc)

                mask = np.asarray(np.greater(mask, thresh), dtype=np.float)

                # Blurring in kspace
                sigma = 2 * np.random.rand()
                mask *= np.exp(-radius * sigma)
                ksp2 = ksp[s] * mask  # (coil, h, w)

                # Translational motion in kspace
                ksp2 = trans_motion(ksp2, 2, 40, 5)  # (input, mode, maxshift, prob)
                ksp2_gpu = sp.to_device(ksp2, device=device)

                #  RECON and diff
                for lamda in np.geomspace(1e-20, 1e-7, num=14):

                    # image_sq = get_l1wavelet(ksp2_gpu, smaps, lamda=lamda, num_iter=20)
                    image_sq = get_l1tv(ksp2_gpu, smaps, lamda=lamda, num_iter=300)
                    error = mse(image_sq, image_sense)
                    diff.append(error)



    if index_file > 0:
        break

diff_tot = np.asarray(diff)
lam = np.geomspace(1e-20, 1e-7, num=14)
plt.semilogx(lam,diff_tot)
plt.grid(True, color='0.7', linestyle='-', which='both', axis='both')
plt.show()
