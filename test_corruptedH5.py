import h5py
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
import logging

import sigpy as sp
import sigpy.mri as mri
import sigpy.plot as pl
import fnmatch
import os
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()
filepath = tk.filedialog.askdirectory(title='Select NYU.h5 folder')
os.chdir(filepath)
file = os.path.join(filepath, 'TRAINING_IMAGES_v5.h5')

hf = h5py.File(name=file, mode='r')

# Make a real database of images
X_T = np.zeros((8,320,320),dtype=np.complex128)
X_0 = np.zeros((8,320,320),dtype=np.complex128)
X_1 = np.zeros((8,320,320),dtype=np.complex128)
for i in range(0, 8):

    name = 'EXAMPLE_%07d_TRUTH' % (i+1)
    X_T[i, :, :] = np.array(hf[name])

    name = 'EXAMPLE_%07d_IMAGE_%04d' % (i + 1, 0)   # 0000.png
    X_0[i, :, :] = np.array(hf[name])

    name = 'EXAMPLE_%07d_IMAGE_%04d' % (i + 1, 1)   # 0001.png
    X_1[i, :, :] = np.array(hf[name])

