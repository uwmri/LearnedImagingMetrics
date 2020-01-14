import h5py
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

import sigpy as sp
import sigpy.mri as mri
import sigpy.plot as pl
import fnmatch
import os
import torch

from fastMRI.data import transforms as T

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                if os.path.islink(name) == False:

                    result.append(os.path.join(root, name))
    return result


train_folder = Path("D:/NYUbrain/brain_multicoil_train/multicoil_train")
val_folder = Path("D:/NYUbrain/brain_multicoil_val/multicoil_val")
test_folder = Path("D:/NYUbrain/brain_multicoil_test/multicoil_test")

files = find("*.h5",train_folder)
file = files[np.random.randint(3000)]


#file = train_folder / "file_brain_AXFLAIR_209_6001413.h5"
#file = val_folder / "file_brain_AXT1_202_6000537.h5"
#file = test_folder / "file_brain_AXFLAIR_201_6002990.h5"


hf = h5py.File(file)
print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))

ksp = hf['kspace'][()]
print(ksp.dtype)
print(ksp.shape)

# mask = hf['mask'][()]

# show a few coils from a random slice
def show_coils(data, coil_nums, cmap=None):
    fig = plt.figure()
    for i, num in enumerate(coil_nums):
        plt.subplot(1, len(coil_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
    plt.show()


# Grab a random slice
nslice = np.random.randint(np.size(ksp,0))
slice_check = ksp[nslice]
coil2 = np.random.randint(np.size(ksp,1))
show_coils(np.log(np.abs(slice_check) + 1e-9), [0, coil2])  # in case of zeros

slice_check2 = T.to_tensor(slice_check)      # Convert from numpy array to pytorch tensor
slice_image = T.ifft2(slice_check2)
slice_image_abs = T.complex_abs(slice_image)

show_coils(slice_image_abs, [0, coil2], cmap='gray')    # show images from the first coil and a random one
plt.show()


# check non-zero ksp elements position
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    each_dim = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    return each_dim.max()


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    each_dim = np.where(mask.any(axis=axis), val, invalid_val)
    return each_dim.max()


ksp_check2d = slice_check[coil2,:,:]
nz_start = first_nonzero(ksp_check2d, axis=1, invalid_val=-1)        # return -1 if all are zeros
nz_end = last_nonzero(ksp_check2d, axis=1, invalid_val=-1)

print(f'Number of non-zeros in PE direction starts at {nz_start} and ends at {nz_end}')


# RECON Sum of squares
slice_image_sos = T.root_sum_of_squares(slice_image_abs, dim=0)
plt.imshow(np.abs(slice_image_sos.numpy()), cmap='gray')
plt.show()      # image is 640 by 320

# Crop 640 direction to 320
slice_image_sos_crop = slice_image_sos[160:480,:]
plt.imshow(np.abs(slice_image_sos_crop.numpy()), cmap='gray')
plt.show()




# smaps ESPIRiT
device = sp.Device(0)
slice_check_gpu = sp.to_device(slice_check,device=device)

smaps = sp.mri.app.EspiritCalib(slice_check_gpu, calib_width=24, thresh=0.005, kernel_width=7, crop=0.8, max_iter=50,
                                            device=device, show_pbar=True).run()        # (coil, h, w)
pl.ImagePlot(smaps, z=0, title='Sensitivity Maps Estimated by ESPIRiT')


# SENSE RECON (l2)
lamda = 0.005
slice_image_sense = mri.app.SenseRecon(slice_check_gpu, smaps, lamda=lamda,device=device, max_iter=30).run()

# crop and flip up down
slice_image_sense = slice_image_sense[160:480,:]
slice_image_sense = sp.to_device(slice_image_sense, sp.cpu_device)
plt.imshow(np.abs(slice_image_sense), cmap='gray')
plt.show()



