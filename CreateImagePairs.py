import h5py
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
import cv2

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


def generate_pairs():
    train_folder = Path("D:/NYUbrain/brain_multicoil_train/multicoil_train")


    val_folder = Path("D:/NYUbrain/brain_multicoil_val/multicoil_val")
    test_folder = Path("D:/NYUbrain/brain_multicoil_test/multicoil_test")

    train_folder_5 = Path("D:/NYUbrain/brain_multicoil_train_5")

    files = find("*.h5", train_folder_5)  # list of file paths

    device = sp.Device(0)

    count = 1

    for index_file, file in enumerate(files):
        print('loading', file)

        hf = h5py.File(file)

        ksp = hf['kspace'][()]
        header = hf['ismrmrd_header'][()]
        ksp_width = ksp.shape[3]
        ksp_height = ksp.shape[2]

        xv, yv = np.meshgrid(np.linspace(-1, 1, ksp_height), np.linspace(-1, 1, ksp_width), sparse=False, indexing='ij')
        radius = np.sqrt(np.square(xv) + np.square(yv))
        print(f'radius shape is {radius.shape}')

        # Number of slices to grab per h5 file
        num_slices = 4
        tot_slices = np.size(ksp, 0)

        # # show sos recon of all slices
        # for i in range(tot_slices):
        #     images_complex = T.ifft2(T.to_tensor(ksp[i]))
        #     images_abs = T.complex_abs(images_complex)
        #     image_sos = T.root_sum_of_squares(images_abs,dim=0)
        #
        #     plt.imshow(np.abs(image_sos.numpy()), cmap='gray')
        #     plt.show()

        # Get list of slice numbers without duplicates
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

        # for each slice, generate truth, corrupted1, corrupted2
        for s in slice_nums:
            s = int(s)

            acc_base = np.random.randint(0, 10)

            for recon in range(0, 2):

                # TRUTH by espirit
                if recon == 0:
                    ksp_gpu = sp.to_device(ksp[s], device=device)  # (coil, h, w)
                    smaps = sp.mri.app.EspiritCalib(ksp_gpu, calib_width=24, thresh=0.005, kernel_width=7, crop=0.8,
                                                    max_iter=50,
                                                    device=device, show_pbar=True).run()
                    lamda = 0.005
                    image_sense = mri.app.SenseRecon(ksp_gpu, smaps, lamda=lamda, device=device,
                                                     max_iter=30).run()
                    # crop and flip up down
                    width = image_sense.shape[1]
                    if width < 320:
                        image_sense = image_sense[160:480, :]
                    else:
                        image_sense = image_sense[.5 * width:1.5 * width, :]

                    # send to cpu and normalize
                    image_sense = sp.to_device(image_sense, sp.cpu_device)
                    image_sense = np.flipud(image_sense)
                    im = Image.fromarray(255 * np.abs(image_sense) / np.max(np.abs(image_sense)))
                    im = im.convert("L")
                    # save
                    name = 'NYU_%07d_TRUTH.png' % (count)
                    name = os.path.join('Rank_NYU', 'ImagePairs_png', name)
                    im.save(name)

                    print(f'saving slice {s} of file # {index_file + 1} to count {count}')

                # Create a sampling mask
                mask = np.random.random_sample(ksp[s,0,:,:].shape)
                variable_density = np.random.rand()
                sampling_density = np.ones(ksp[s,0,:,:].shape) * variable_density + (1 - variable_density) / (0.01 + radius)
                mask *= sampling_density

                acc = acc_base + np.random.randint(0, 10)

                thresh = np.percentile(mask, acc)

                mask = np.asarray(np.greater(mask, thresh), dtype=np.float)

                # Blurring in kspace
                sigma = 2 * np.random.rand()
                mask *= np.exp(-radius * sigma)
                ksp2 = ksp[s] * mask
                print(f'ksp2 shape is {ksp2.shape}')

                # RECON
                ksp_gpu2 = sp.to_device(ksp2, device=device)  # (coil, h, w)
                smaps2 = sp.mri.app.EspiritCalib(ksp_gpu2, calib_width=24, thresh=0.005, kernel_width=7, crop=0.8,
                                                 max_iter=50,
                                                 device=device, show_pbar=True).run()
                lamda = 0.005
                image_sense2 = mri.app.SenseRecon(ksp_gpu2, smaps2, lamda=lamda, device=device,
                                                  max_iter=30).run()
                # crop and flip up down
                width = image_sense2.shape[1]
                if width < 320:
                    image_sense2 = image_sense2[160:480, :]
                else:
                    image_sense2 = image_sense2[.5 * width:1.5 * width, :]

                # send to cpu and normalize
                image_sense2 = sp.to_device(image_sense2, sp.cpu_device)
                image_sense2 = np.flipud(image_sense2)

                ksp3 = np.fft.fftshift(np.fft.fft2(image_sense2))
                print(f'ksp3 shape is {ksp3.shape}')
                (num_rows, num_cols) = image_sense2.shape
                #print(f'image_sense2 shape is {image_sense2.shape}')

                # Add translational motion in image domain, columns are phase encode
                maxshift = 1
                for jj in range(num_cols):
                    if np.random.randint(10) == 1:
                        translation_matrix = np.float32([[1, 0, np.random.uniform(0, maxshift)], [0, 1, 0]])
                        image2_real = cv2.warpAffine(np.real(image_sense2), translation_matrix, (num_cols, num_rows))
                        image2_img = cv2.warpAffine(np.imag(image_sense2), translation_matrix, (num_cols, num_rows))
                        image_sense2 = image2_real + 1j * image2_img
                    else:
                        pass

                    kspace_temp = np.fft.fftshift(np.fft.fft2(image_sense2))

                    #print(f'kspace_temp shape is {kspace_temp.shape}')
                    ksp3[:, jj] = kspace_temp[:, jj]

                image_sense2 = np.fft.ifft2(np.fft.fftshift(ksp3))

                # Wavelet Compression as in Compressed Sensing
                image_sense2 = wave_thresh(image_sense2, 0.9 + 0.1 * np.random.rand())

                # Find scaling that minimizes L2 Loss
                scale = np.sum(image_sense2 * np.conj(image_sense)) / np.sum(image_sense * np.conj(image_sense))
                print(f'scale for file # {index_file}, slice # {s}, recon # {recon} is {scale} ')
                image_sense2 *= scale

                if 1 == 1:
                    plt.figure
                    plt.subplot(131)
                    plt.imshow(np.abs(image_sense))
                    plt.subplot(132)
                    plt.imshow(np.abs(mask))
                    plt.subplot(133)
                    plt.imshow(np.abs(image_sense2))
                    plt.show()

                # save png
                name = 'NYU_%07d_IMAGE_%04d.png' % (count, recon)
                name = os.path.join('Rank_NYU', 'ImagePairs_png', name)
                im = Image.fromarray(255 * np.abs(image_sense2) / np.max(np.abs(image_sense2)))
                im = im.convert("L")
                im.save(name)

            count += 1


def soft_thresh(input, thresh):
    sort_input = np.sort(np.abs(input.flatten()))
    idx = int(sort_input.size - np.floor(sort_input.size * thresh))

    thresh = sort_input[idx]
    output = sp.thresh.soft_thresh(thresh, input)

    return output


def wave_thresh(input, thresh):
    W = sp.linop.Wavelet(input.shape, wave_name='db4')
    output = W(input)
    output = soft_thresh(output, thresh)

    output = W.H(output)
    return output


if __name__ == "__main__":
    generate_pairs()
