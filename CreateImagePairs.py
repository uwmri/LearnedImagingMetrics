import h5py
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
import logging
from skimage.metrics import structural_similarity
from skimage.measure import compare_nrmse

import sigpy as sp
import sigpy.mri as mri
import sigpy.plot as pl
import fnmatch
import os
import torch
import pywt
import time

from fastMRI.data import transforms as T



def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                if os.path.islink(name) == False:
                    result.append(os.path.join(root, name))
    return result

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


def trans_motion(input, dir_motion, maxshift, sigma_shift, prob):
    # input: kspace (coil, h, w).
    # dir_motion: direction of phase encode. 0 for width, 1 for height-dir
    # maxshift: np.random.uniform(0,maxshift) is the size of displacement in pixel
    # prob: if np.random.randint(prob) == 1, there's motion for that PE line

    logger = logging.getLogger('trans_motion')

    if dir_motion == 0:
        num_pe = input.shape[2]     # number of column
        for ii in range(num_pe):
            if np.random.randint(prob) == 0:
                input[:, :, ii] = input[:, :, ii] * np.exp(
                    -1j * 2 * np.pi * np.random.uniform(0, maxshift) * (1 / num_pe) * (ii - num_pe / 2))
            else:
                pass
    elif dir_motion == 1:
        num_pe = input.shape[1]     # number of rows
        for jj in range(num_pe):
            if np.random.randint(prob) == 0:
                input[:, jj, :] = input[:, jj, :] * np.exp(
                    -1j * 2 * np.pi * np.random.uniform(0, maxshift) * (1 / num_pe) * (jj - num_pe / 2))
            else:
                pass
    else:
        # Motion in both directions. Continuous fashion.
        # The whole thing move a fixed amount. Not doing per line random amount like above
        logger.info(f'probability of motion is 1/{prob}')
        if np.random.randint(prob) == 0:

            logger.info(f'Yes, Motion')
            num_x, num_y = input.shape[1:]
            startdir = np.random.randint(0,1)   # Always start in PE (column)

            shift_bulk0 = (2 * np.random.randint(0, 2) - 1) * np.random.randint(1, maxshift)  # shift in x
            shift_bulk1 = (2 * np.random.randint(0, 2) - 1) * np.random.randint(1, maxshift)  # shift in y
            # shift_bulk0 = (2*np.random.randint(0,2)-1) * np.random.normal(maxshift, sigma_shift)        # shift in x
            # shift_bulk1 = (2*np.random.randint(0,2)-1) * np.random.normal(maxshift, sigma_shift)        # shift in y


            logger.info(f'shift in x is {shift_bulk0} pixels')
            logger.info(f'shift in y is {shift_bulk1} pixels')

            if startdir == 1:
                start = np.random.randint(0,num_x)
                logger.info(f'motion starts from row # {start}')

                for ii in range(start,num_x):
                    for jj in range(num_y):
                        input[:,ii, jj] = input[:, ii, jj] * np.exp(
                            -1j * 2 * np.pi * (shift_bulk0 * (1 / num_y) * (jj - num_y / 2) + shift_bulk1 * (1 / num_x)
                                               * (ii - num_x / 2)))
            else:   # col is PE
                mu_start = num_y/2
                sigma_start = num_y/2*.075
                start = np.int(np.floor(np.random.normal(mu_start,sigma_start)))
                central_rangeL = num_y/2 - num_y/2*.05
                central_rangeR = num_y/2 + num_y/2*.05
                if start > central_rangeR:

                    logger.info(f'motion starts from column # {start} to the end')
                    for ii in range(num_x):
                        for jj in range(start,num_y):
                            input[:, ii, jj] = input[:, ii, jj] * np.exp(
                                -1j * 2 * np.pi * (shift_bulk0 * (1 / num_y) * (jj - num_y / 2) + shift_bulk1 *
                                                   (1 / num_x) * (ii - num_x / 2)))
                elif start < central_rangeL:

                    logger.info(f'motion starts from column #0 to {start}')
                    for ii in range(num_x):
                        for jj in range(0,start):
                            input[:, ii, jj] = input[:, ii, jj] * np.exp(
                                -1j * 2 * np.pi * (-shift_bulk0 * (1 / num_y) * (jj - num_y / 2) - shift_bulk1 *
                                                   (1 / num_x) * (ii - num_x / 2)))
                else:

                    logger.info(f'motion starts too close to the central region, abort')
        else:
            pass

    return input


def normalization(input, num_max):
    # normalize to the mean of the biggest N elements
    temp = np.partition(-np.abs(input), num_max)
    norm_fac = np.mean(-temp[:num_max])
    return np.abs(input)/norm_fac


def add_gaussian_noise(input, prob, median_real=None, median_imag=None, level=1, mode=0,mean=0):
    # input: when mode=0, it's image before crop to square, (h,w) complex. When mode=1,its ksp2 = (coil, h, w)
    # level: the sigma of gaussian noise = level * median(HH)
    # mode: 0 is to take median(hh) as sigma. 1 is to take edge of ksp as sigma.
    # median: only use when mode=1. it is median|kedge|. dim = (coil,)

    # add to image
    logger = logging.getLogger('add_gaussian_noise')

    if mode == 0:
        coeff = pywt.wavedec2(input, 'db4', mode='per', level=1)
        ll,(lh,hl,hh) = coeff
        sigma_real = level * np.abs(np.median(np.real(hh)))
        sigma_imag = level * np.abs(np.median(np.imag(hh)))

        height, width = input.shape

        if np.random.randint(prob) == 0:

            gauss_real = np.random.normal(mean, sigma_real, (height, width))
            gauss_real = gauss_real.reshape(height, width)

            gauss_imag = np.random.normal(mean, sigma_imag, (height, width))
            gauss_imag = gauss_imag.reshape(height, width)

            logger.info(f'zero mean Gaussian noise, sigma_real = {sigma_real}, sigma_imag = {sigma_imag}')

            noisy_real = np.real(input) + gauss_real
            noisy_imag = np.imag(input) + gauss_imag
            noisy = noisy_real + 1j*noisy_imag
        else:
            logger.info('No gaussian noise added')
            noisy = input
    else:
        # add to ksp2
        height = input.shape[1]
        width = input.shape[2]
        if np.random.randint(prob) == 0:
            noisy=[]
            logger.info('Gaussian noise added to kspace(coil, h,w)')
            logger.info(f'Gaussian noise level = {level}')
            for c in range(input.shape[0]):
                sigma_real = level * median_real[c]
                sigma_imag = level * median_imag[c]

                gauss_real = np.random.normal(mean, sigma_real, (height, width))
                gauss_real = gauss_real.reshape(height, width)

                gauss_imag = np.random.normal(mean, sigma_imag, (height, width))
                gauss_imag = gauss_imag.reshape(height, width)

                #logger.info(f'zero mean Gaussian noise, sigma_real = {sigma_real}, sigma_imag = {sigma_imag}')

                noisy_real = np.real(input[c]) + gauss_real
                noisy_imag = np.imag(input[c]) + gauss_imag
                noisy_c = noisy_real + 1j*noisy_imag
                noisy.append(noisy_c)
            noisy = np.asarray(noisy)
        else:
            logger.info('No gaussian noise added')
            noisy = input

    return noisy


def add_incoherent_noise(kspace=None, prob=None, central=0.4, mode=1, num_corrupted=0, dump=0.5):
    # feed in ksp(coil, h, w)
    # baseline is uniform random of 1 and 0
    # percent: percentage of ones and (1-1/prob) zeros
    # mode: 0 is to drop random ksp points, 1 is to drop random ksp columns (PE)
    # central - (1-central) will never be discarded

    global percent
    logger = logging.getLogger('add_incoherent_noise')

    kspace_width = kspace.shape[2]
    centralL = np.int(np.floor(kspace_width * central))
    centralR = np.int(np.floor(kspace_width * (1-central)))

    if np.random.randint(prob) == 0:
        logger.info('Incoherent noise added')
        if mode == 1:
            if num_corrupted == 0:
                percent = np.random.uniform(dump,1)
            else:
                # for the second corrupted image, do the same the percent as the first one. input dump=percent1.
                percent = dump
            logger.info(f'mode={mode}, discard {(1-percent)*100}% PEs')
            randuni_col = np.random.choice([0,1], size=kspace[0, 0,:].shape, p=[1-percent, percent])

            randuni_colm = np.tile(randuni_col, (len(kspace[0, :, 0]),1))
            randuni_colm[:, centralL:centralR] = 1
            for c in range(len(kspace[:,0,0])):
                kspace[c, :, :] = kspace[c, :, :] * randuni_colm

        else:
            if num_corrupted == 0:
                dump0 = dump + 0.2
                percent = np.random.uniform(dump0,1)     #default is (0.7, 1), more moderate when discarding points
            else:
                # for the second corrupted image, do the same the percent as the first one. input dump=percent1.
                percent = dump
                print(f'percent for mode 0 is {percent}')
            logger.info(f'mode={mode}, discard {(1 - percent)*100}% points')
            randuni_m = np.random.choice([0, 1], size=kspace[0, :, :].shape, p=[1-percent, percent])

            for c in range(len(kspace[:,0,0])):
                kspace[c, :, :] = kspace[c, :, :] * randuni_m
    else:
        mode = 0
        percent = 0.99
        logger.info('No incoherent noise added')
    return kspace, percent, mode


def get_smaps(kspace_gpu, device, maxiter=50):

    mps = sp.mri.app.EspiritCalib(kspace_gpu, calib_width=24, thresh=0.005, kernel_width=7, crop=0.8,
                                    max_iter=maxiter,
                                    device=device, show_pbar=True).run()
    return mps


def get_truth(kspace, sl, device, lamda):
    kspace_sl = kspace[sl]
    ksp_gpu = sp.to_device(kspace_sl, device=device)  # (coil, h, w)
    smaps = get_smaps(ksp_gpu, device=device, maxiter=50)
    image_truth = mri.app.SenseRecon(ksp_gpu, smaps, lamda=lamda, device=device,
                                     max_iter=20).run()
    # crop and flip up down
    width = image_truth.shape[1]
    if width < 320:
        image_truth = image_truth[160:480, :]
    else:
        image_truth = image_truth[.5 * width:1.5 * width, :]

    # send to cpu and normalize
    image_truth = sp.to_device(image_truth, sp.cpu_device)
    image_truth = np.flipud(image_truth)
    image_truth /= np.max(np.abs(image_truth))

    image_truth = image_truth.astype(np.complex128)
    return image_truth


def get_corrupted(kspace, sl, num_coils, num_corrupted, device, acc=0, acc_ulim=15,
                  kedge_len=30, gaussian_ulim=12, gaussian_prob=2, dir_motion=2, maxshift=20, sigma_shift=6,
                  motion_prob=3, incoherent_prob=2, dump=0.7, mode_incoherent=1):
    logger = logging.getLogger('get_corrupted')

    # get smaps from original ksp
    ksp_full = kspace[sl]
    ksp_full_gpu = sp.to_device(ksp_full, device=device)  # (coil, h, w)
    smaps = get_smaps(ksp_full_gpu, device=device, maxiter=50)
    smaps = sp.to_device(smaps, device=device)
    xv, yv = np.meshgrid(np.linspace(-1, 1, kspace.shape[2]), np.linspace(-1, 1, kspace.shape[3]), sparse=False,
                         indexing='ij')
    radius = np.sqrt(np.square(xv) + np.square(yv))

    # Create a sampling mask
    mask = np.random.random_sample(kspace[sl, 0, :, :].shape)
    variable_density = np.random.rand()
    sampling_density = np.ones(kspace[sl, 0, :, :].shape) * variable_density + (1 - variable_density) / (0.01 + radius)
    mask *= sampling_density

    if num_corrupted == 0:
        acc = np.random.randint(0, acc_ulim)
        logger.info(f'acceleration is {acc} percent')
    else:
        acc = acc
        logger.info(f'acceleration is {acc} percent')

    thresh = np.percentile(mask, acc)
    mask = np.asarray(np.greater(mask, thresh), dtype=np.float)

    # Blurring in kspace
    sigma = 2 * np.random.rand()
    mask *= np.exp(-radius * sigma)
    # ksp2 = ksp[s]        # (coil, h, w)
    ksp2 = np.copy(kspace[sl])

    for c in range(num_coils):
        ksp2[c, :, :] = ksp2[c, :, :] * mask  # ksp2 = (coil, h,w)

    # Using edge of ksp as sigma_estimated
    idx_1nz = next((i for i, x in enumerate(ksp2[0, 0, :]) if x), None)  # index of first non-zero value
    idx_enz = kspace.shape[3] - idx_1nz + 1  # index of last non-zero value

    kedgel = ksp2[:, :, idx_1nz:idx_1nz + kedge_len]
    kedge = np.concatenate((kedgel, ksp2[:, :, idx_enz - kedge_len:idx_enz]), axis=2)
    kedge_real = np.real(kedge)
    kedge_imag = np.imag(kedge)

    sigma_est_real = np.median(np.median(np.abs(kedge_real), axis=1), axis=1)
    sigma_est_imag = np.median(np.median(np.abs(kedge_imag), axis=1), axis=1)

    # Add Gaussian noise to each coil in kspace
    if num_corrupted == 0:
        gaussian_level = np.random.randint(0, gaussian_ulim)
    else:
        gaussian_level = gaussian_ulim      # need to put gaussian_level(recon=0) as gaussian_ulim for later tries

    ksp2 = add_gaussian_noise(ksp2, gaussian_prob, median_real=sigma_est_real, median_imag=sigma_est_imag,
                              level=gaussian_level, mode=1, mean=0)

    # Translational motion in kspace

    ksp2 = trans_motion(ksp2, dir_motion, maxshift, sigma_shift, motion_prob)




    # Add incoherent noise
    if num_corrupted == 0:
        mode = np.ndarray.item(np.random.choice([0, 1], size=1, p=[.8, .2]))  # leaning towards dropping points
    else:
        mode = mode_incoherent
    ksp2, percent, mode = add_incoherent_noise(ksp2, prob=incoherent_prob, central=np.random.uniform(0.2, 0.4), mode=mode, num_corrupted=num_corrupted, dump=dump)
    ksp2_gpu = sp.to_device(ksp2, device=device)

    # RECON. 0 for sos, 1 for PILS, 2 for L2 SENSE, 3 for l1 wavelet, 4 for tv
    recon_type = 0
    # recon_type = np.random.randint(5)

    mu_iter, sigma_iter = 30, 3
    maxiter = np.int(np.ceil(np.abs(np.random.normal(mu_iter, sigma_iter, 1))))

    if recon_type == 0:
        logger.info('SOS Recon')
        ksp2_tensor = T.to_tensor(ksp2)  # T requires array to be tensor
        image = T.ifft2(ksp2_tensor)  # torch.Size([20, 640, 320, 2])
        image = image.numpy()
        image = image[:, :, :, 0] + 1j * image[:, :, :, 1]  # (coil, h, w)
        #abs
        image = np.abs(image)
        image = image.astype('complex128')
        image = np.sqrt(np.sum(image ** 2, axis=0))  # (h, w)

    elif recon_type == 1:
        logger.info('PILS')
        lamda = np.random.uniform(0.001, 0.1)
        image = mri.app.SenseRecon(ksp2_gpu, smaps, lamda=lamda, device=device,
                                   max_iter=1).run()
        image = sp.to_device(image, sp.cpu_device)

    elif recon_type == 2:
        lamda = np.random.uniform(0.01, 100)
        logger.info(f'L2 with {maxiter} iterations, lambda={lamda}')
        image = mri.app.SenseRecon(ksp2_gpu, smaps, lamda=lamda, device=device,
                                   max_iter=maxiter).run()
        image = sp.to_device(image, sp.cpu_device)

    elif recon_type == 3:
        lamda = np.random.uniform(1e-8, 1e-5)
        logger.info(f'L1Wavelet with {maxiter} iterations, lambda={lamda}')
        image = mri.app.L1WaveletRecon(ksp2_gpu, smaps, lamda=lamda, device=device,
                                       max_iter=maxiter).run()
        image = sp.to_device(image, sp.cpu_device)

    elif recon_type == 4:
        maxiterL1TV = 2 * maxiter
        lamda = np.random.uniform(1e-7, 1e-5)
        logger.info(f'L1TV with {maxiter} iterations, lambda={lamda}')
        image = mri.app.TotalVariationRecon(ksp2_gpu, smaps, lamda=lamda, device=device, max_iter=maxiterL1TV).run()
        image = sp.to_device(image, sp.cpu_device)

    # Add Gaussian noise
    # image = add_gaussian_noise(image, 1, level=1000, mean=0)

    # crop and flip up down
    width = image.shape[1]
    if width < 320:
        image_sq = image[160:480, :]
    else:
        lower = int(.5 * width)
        upper = int(1.5 * width)
        image_sq = image[lower:upper, :]

    image_sq = np.flipud(image_sq)

    # Wavelet Compression as in Compressed Sensing
    image_sq = wave_thresh(image_sq, 0.9 + 0.1 * np.random.rand())
    # Normalize
    image_sq /= np.max(np.abs(image_sq))

    return image_sq, acc, gaussian_level, percent, mode    # percent and mode are incoherent noise

def generate_pairs():
    train_folder = Path("D:/NYUbrain/brain_multicoil_train/multicoil_train")

    val_folder = Path("D:/NYUbrain/brain_multicoil_val/multicoil_val")
    test_folder = Path("D:/NYUbrain/brain_multicoil_test/multicoil_test")

    train_folder_5 = Path("D:/NYUbrain/brain_multicoil_train_5")


    files = find("*.h5", train_folder_5)  # list of file paths

    device = sp.Device(0)

    count = 1

    logger = logging.getLogger('generate_pairs')

    # Export to hdf5
    out_name = os.path.join('TRAINING_IMAGES_v5.h5')
    try:
        os.remove(out_name)
    except OSError:
        pass

    for index_file, file in enumerate(files):
        logger.info(f'loading {file}')

        # get rid of files with less than 8 coils
        file_size = os.path.getsize(file)
        if file_size < 300000000:
            pass
        else:
            hf = h5py.File(file)

            ksp = hf['kspace'][()]

            # Number of slices to grab per h5 file
            num_slices = 4
            tot_slices = np.size(ksp, 0)
            tot_coils = np.size(ksp, 1)

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

                # Get Truth
                image_sense = get_truth(ksp, s, device=device, lamda=0.005)    # has been normalized to its max, still complex

                # save
                im = Image.fromarray(255 * np.abs(image_sense))
                im = im.convert("L")
                name = 'NYU_%07d_TRUTH.png' % count
                name = os.path.join('Rank_NYU', 'ImagePairs_png_v6', name)
                im.save(name)

                print(f'saving slice {s} of file # {index_file + 1} to count {count}')
                logger.info('...')
                logger.info(f'saving slice {s} to count {count}')

                with h5py.File(out_name, 'a') as hf:
                    name = 'EXAMPLE_%07d_TRUTH' % count
                    hf.create_dataset(name, data=image_sense)

                # Get Corrupted
                for recon in range(0, 2):

                    if recon == 0:

                        image_corrupted1, acc1, gaussian_level1, percent1, mode1 = get_corrupted(ksp, s, num_corrupted=recon, num_coils=tot_coils, device=device)  # has been normalized to its max
                        logger.info(f'gaussian_level is {gaussian_level1} for both')
                        # Find scaling that minimizes MSE(corrupted, ori)
                        scale = np.sum(np.conj(image_corrupted1).T * image_sense) / np.sum(
                            np.conj(image_corrupted1).T * image_corrupted1)
                        # print(f'scale for file # {index_file + 1}, slice # {s}, recon # {recon} is {scale} ')
                        # logger.info(f'scale is {scale}')
                        image_corrupted1 *= scale

                        # mse1 = compare_nrmse(image_sense, image_corrupted1, norm_type='euclidean')
                        ssim1 = structural_similarity(np.abs(image_corrupted1), np.abs(image_sense))
                        # logger.info(f'mse between corrupted1 and truth = {mse1}')
                        logger.info(f'SSIM between corrupted1 and truth = {ssim1}')

                        name = 'NYU_%07d_IMAGE_%04d.png' % (count, recon)
                        name = os.path.join('Rank_NYU', 'ImagePairs_png_v6', name)
                        im = Image.fromarray(255 * np.abs(image_corrupted1))
                        im = im.convert("L")
                        im.save(name)
                        logger.info(f'saving to png {name}')
                        logger.info('...')

                        with h5py.File(out_name, mode='a') as hf:
                            name = 'EXAMPLE_%07d_IMAGE_%04d' % (count, recon)
                            hf.create_dataset(name, data=image_corrupted1)

                    else:
                        image_corrupted, acc, gaussian_level, percent, mode = get_corrupted(ksp, s, num_corrupted=recon, num_coils=tot_coils, device=device, acc=acc1, gaussian_ulim=gaussian_level1, dump=percent1, mode_incoherent=mode1)  # has been normalized to its max
                        logger.info(f'Incoherent noise mode is {mode}. dump {1-percent} PE for addding incoherent noise')
                        # Find scaling that minimizes MSE(corrupted, ori)
                        scale = np.sum(np.conj(image_corrupted).T * image_sense) / np.sum(
                            np.conj(image_corrupted).T * image_corrupted)
                        # print(f'scale for file # {index_file + 1}, slice # {s}, recon # {recon} is {scale} ')
                        # logger.info(f'scale is {scale}')
                        image_corrupted *= scale

                        # mse2 = compare_nrmse(image_sense, image_corrupted, norm_type='euclidean')

                        ssim2 = structural_similarity(np.abs(image_corrupted), np.abs(image_sense))
                        diff_ssim12 = np.abs(ssim2 - ssim1)
                        # logger.info(f'mse between corrupted2 and truth = {mse2}')
                        logger.info(f'SSIM between corrupted2 and truth = {ssim2}')
                        logger.info(f'SSIM between corrupted1 and 2 = {diff_ssim12}')
                        counter_regenerate = 1

                        # while np.abs(mse2-mse1) > 0.07 or np.abs(ssim2-ssim1) > 0.07:

                        while diff_ssim12 > 0.09:
                            logger.info(f'ssim too large, regenerate')
                            image_corrupted, acc, gaussian_level, percent, mode = get_corrupted(ksp, s, num_corrupted=recon, num_coils=tot_coils,
                                                                 device=device,
                                                                 acc=acc1, gaussian_ulim=gaussian_level1, dump=percent1,
                                                                                                mode_incoherent=mode1)  # has been normalized to its max
                            scale = np.sum(np.conj(image_corrupted).T * image_sense) / np.sum(
                                np.conj(image_corrupted).T * image_corrupted)
                            image_corrupted *= scale

                            # mse2 = compare_nrmse(image_sense, image_corrupted, norm_type='euclidean')
                            ssim2 = structural_similarity(np.abs(image_corrupted), np.abs(image_sense))
                            diff_ssim12 = np.abs(ssim2 - ssim1)

                            counter_regenerate += 1
                            if counter_regenerate > 12:
                                logger.info(f'Too many tries, settle on this one. ssim12 = {np.abs(ssim2-ssim1)}')
                                name = 'NYU_%07d_IMAGE_%04d.png' % (count, recon)
                                name = os.path.join('Rank_NYU', 'ImagePairs_png_v6', name)
                                im = Image.fromarray(255 * np.abs(image_corrupted))
                                im = im.convert("L")
                                im.save(name)
                                logger.info(f'saving to png {name}')
                                logger.info('...')

                                with h5py.File(out_name, mode='a') as hf:
                                    name = 'EXAMPLE_%07d_IMAGE_%04d' % (count, recon)
                                    hf.create_dataset(name, data=image_corrupted)

                                break
                        else:
                            logger.info(f'SSIM between corrupted1 and 2 = {diff_ssim12}')
                            name = 'NYU_%07d_IMAGE_%04d.png' % (count, recon)
                            name = os.path.join('Rank_NYU', 'ImagePairs_png_v6', name)
                            im = Image.fromarray(255 * np.abs(image_corrupted))
                            im = im.convert("L")
                            im.save(name)
                            logger.info(f'saving to png {name}')
                            logger.info('...')

                            with h5py.File(out_name, mode='a') as hf:
                                name = 'EXAMPLE_%07d_IMAGE_%04d' % (count, recon)
                                hf.create_dataset(name, data=image_corrupted)

                if 1 == 0:
                    plt.figure()
                    plt.subplot(131)
                    plt.imshow(np.abs(image_sense))
                    plt.subplot(132)
                    plt.imshow(np.abs(image_corrupted1))
                    plt.subplot(133)
                    plt.imshow(np.abs(image_corrupted))
                    plt.show()
                    plt.close()

                count += 1

            if index_file > 4:
                break


if __name__ == "__main__":

    logging.basicConfig(filename='CreateImagePairs_NYU.log', filemode='w', level=logging.INFO)
    logger = logging.getLogger('main')

    start_time = time.time()
    generate_pairs()

    total_time = (time.time() - start_time)/60

    print("--- %s min ---" % total_time)
