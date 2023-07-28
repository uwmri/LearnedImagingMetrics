import cupy as cp
import numpy as np
import sigpy as sp
import math
import torch


mod_cuda = """
__device__ inline int mod(int x, int n) {
    return (x % n + n) % n;
}
"""

image_modulation = cp.ElementwiseKernel(
    'T im, raw T mod_amount','T im_mod', """    
    const int width = mod_amount.shape()[-1];
    const int idx = mod(i, width);    
    im_mod = (T) im * mod_amount[idx];
    """,
    'image_modulation',
    preamble=mod_cuda
)

def add_phase_im(image, kshift):
    """
    Add linear phase in PE direction by modulating the image.
    Note that the added phase cannpt be undone precisely
    when the dataset is zero-padded.
    :param image: (ch, h, w) cupy array
    :param kshift_max: in pixel
    :return: image (ch, h, w) cupy array
    """
    # kshift = np.random.randint(-kshift_max, kshift_max)
    num_PE = image.shape[-1]
    mod_amount = cp.exp(1j * 2 * np.pi * kshift * (cp.arange(0,num_PE) - num_PE//2) / num_PE)
    mod_amount = mod_amount.astype(image.dtype)
    # for ii in range(num_PE):
    #     im_addedPhase[:,:,ii] = image[:,:,ii] * np.exp(1j * 2 * np.pi * kshift * (ii - num_PE//2) / num_PE)
    im_flat = image.flatten()
    im_addedPhase = image_modulation(im_flat, mod_amount)
    im_addedPhase = im_addedPhase.reshape(image.shape)
    return im_addedPhase


def sigpy_image_rotate2(image, theta, verbose=False, device=sp.Device(0)):
    """
    SIgpy based 2D image rotation
    Args:
        image (list or array): List of images to rotate, size [ch,nx,nx]
        theta(float): image rotation
    """
    device = sp.Device(device)
    xp = device.xp

    with device:

        # Random rotation
        r = xp.array([[math.cos(theta), math.sin(theta)],
                     [-math.sin(theta), math.cos(theta)]], np.float32)

        if verbose:
            print(f'Rotation = {r}')

        if isinstance(image, list):
            ny = image[0].shape[1]
            nx = image[0].shape[2]
            nch = image[0].shape[0]
        else:
            ny = image.shape[1]
            nx = image.shape[2]
            nch = image.shape[0]

        # Pad to be account for zero padding
        pad = int( (nx*0.42) // 2)

        ny_pad = ny + 2 * pad
        nx_pad = nx + 2 * pad

        cx = nx // 2
        cy = ny // 2
        cx_pad = nx_pad // 2
        cy_pad = ny_pad // 2

        # Rotate the image coordinates
        y, x = xp.meshgrid(xp.arange(0, ny, dtype=xp.float32),
                           xp.arange(0, nx, dtype=xp.float32),
                           indexing='ij')

        #subtract center of coordinates
        x -= cx
        y -= cx

        # Rotate the coordinates
        coord = xp.stack((y, x), axis=-1)

        if verbose:
            print(f'Coord Shape {coord.shape}')

        coord = xp.expand_dims(coord, -1)
        coord = xp.matmul(r, coord)
        coord = xp.squeeze(coord)
        coord[:, :, 0] += cx_pad
        coord[:, :, 1] += cy_pad


        # Actual rotation
        if isinstance(image, list):
            image_rotated = []
            for imt in image:
                xpad = xp.zeros((ny_pad, nx_pad), dtype=imt.dtype)
                xnew = xp.zeros((nch, ny, nx), dtype=imt.dtype)
                for ch in range(nch):
                    xpad[pad:pad + ny, pad:pad + ny] = imt[ch]
                    xnew[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)
                image_rotated.append(xnew)
        else:
            image_rotated = xp.zeros((nch, ny, nx), dtype=xp.float32)
            xpad = xp.zeros((ny_pad, nx_pad), dtype=xp.float32)
            for ch in range(nch):
                xpad[pad:pad + ny, pad:pad + ny] = image[ch]
                image_rotated[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)

    return image_rotated


def sigpy_image_rotate3( image, theta, verbose=False, device=sp.Device(0), crop=False):
    """
    SIgpy based 2D image rotation

    Args:
        image (list or array): List of images to rotate, size [ch,nx,nx]
        theta(float): image rotation

    """
    device = sp.Device(device)
    xp = device.xp

    with device:

        # Random rotation
        r = xp.array([[math.cos(theta), math.sin(theta)],
                     [-math.sin(theta), math.cos(theta)]], np.float32)

        if verbose:
            print(f'Rotation = {r}')

        if isinstance(image, list):
            ny = image[0].shape[1]
            nx = image[0].shape[2]
            nch = image[0].shape[0]
        else:
            ny = image.shape[1]
            nx = image.shape[2]
            nch = image.shape[0]

        # Pad to be account for zero padding
        pad = int( (nx*0.42) // 2)

        ny_pad = ny + 2 * pad
        nx_pad = nx + 2 * pad

        cx = nx // 2
        cy = ny // 2
        cx_pad = nx_pad // 2
        cy_pad = ny_pad // 2

        # Rotate the image coordinates
        if crop:
            y, x = xp.meshgrid(xp.arange(0, ny_pad, dtype=xp.float32),
                               xp.arange(0, nx_pad, dtype=xp.float32),
                               indexing='ij')
            x -= cx_pad
            y -= cy_pad
        else:
            y, x = xp.meshgrid(xp.arange(0, ny, dtype=xp.float32),
                           xp.arange(0, nx, dtype=xp.float32),
                           indexing='ij')

            #subtract center of coordinates
            x -= cx
            y -= cy

        # Rotate the coordinates
        coord = xp.stack((y, x), axis=-1)

        if verbose:
            print(f'Coord Shape {coord.shape}')

        coord = xp.expand_dims(coord, -1)
        coord = xp.matmul(r, coord)
        coord = xp.squeeze(coord)
        coord[:, :, 0] += cx_pad
        coord[:, :, 1] += cy_pad

        # Actual rotation
        if isinstance(image, list):
            image_rotated = []
            if crop:
                for imt in image:
                    xnew_uncrop = xp.zeros((nch, ny_pad, nx_pad), dtype=xp.float32)
                    xnew = xp.zeros((nch, ny, nx), dtype=xp.float32)
                    for ch in range(nch):
                        xpad = imt[ch]
                        xnew_uncrop[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)
                        xnew[ch] = xnew_uncrop[ch,pad:pad + ny, pad:pad + ny]
                    image_rotated.append(xnew)
            else:
                for imt in image:
                    xpad = xp.zeros((ny_pad, nx_pad), dtype=xp.float32)
                    xnew = xp.zeros((nch, ny, nx), dtype=xp.float32)
                    for ch in range(nch):
                        xpad[pad:pad + ny, pad:pad + ny] = imt[ch]
                        xnew[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)
                    image_rotated.append(xnew)
        else:
            if crop:
                image_rotated_uncrop = xp.zeros((nch, ny_pad, nx_pad), dtype=xp.float32)
                image_rotated = xp.zeros((nch, ny, nx), dtype=xp.float32)
                for ch in range(nch):
                    xpad = image[ch]
                    image_rotated_uncrop[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)
                    image_rotated[ch] = image_rotated_uncrop[ch,pad:pad + ny, pad:pad + ny]
            else:
                image_rotated = xp.zeros((nch, ny, nx), dtype=xp.float32)
                xpad = xp.zeros((ny_pad, nx_pad), dtype=xp.float32)
                for ch in range(nch):
                    xpad[pad:pad + ny, pad:pad + ny] = image[ch]
                    image_rotated[ch] = sp.interpolate(xpad, coord, kernel='spline', width=2)

    return image_rotated
