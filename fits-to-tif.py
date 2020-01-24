#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

import glob
import os
from PIL import Image
import time

import healpy as hp
import astropy
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action='store_true', help="overwrite maps.")
parser.add_argument("--mod", action='store_true', help="apply modification.")
parser.add_argument("--norm", action='store_true', help="apply normalization.")
parser.add_argument("--RGB", action='store_true', help="convert to RGB.")
parser.add_argument("--xsize", type=int, help="xsize of the tif figure.", default=3000)

args = parser.parse_args()
overwrite = args.overwrite
mod = args.mod
norm = args.norm
rgb = args.rgb
x_size = args.xsize

# transform all .fits maps in dircetory into .tif
path = "/archive/home/sammazza/radioML/data/mapsim_lowN/"
destinationPath_data = "/archive/home/sammazza/radioML/data/mapsim_tif_data_lowN/"
destinationPath_label = "/archive/home/sammazza/radioML/data/mapsim_tif_label_lowN/"

all_labels = glob.glob(os.path.join(path, "*.fits")) # changed output name for easier identification

print('There are %i maps in %r'%(len(all_labels), path))

y_size = int(x_size/2)
print('Producing tif maps of size:',x_size,'X',y_size)

####### modification funtions
center = 0.5 # centering data around 0.5 to not have negative values
def modification(moll_array):
    moll_array[np.isinf(moll_array)] = center
    moll_array = moll_array + center
    return moll_array

def toRGB(im):
    rgbArray = np.zeros((im.shape[0], im.shape[1], 3), 'uint8')
    rgbArray[..., 0] = im*256
    rgbArray[..., 1] = im*256
    rgbArray[..., 2] = im*256
    return rgbArray

def normalization(moll_array):
    moll_array = moll_array + np.abs(np.min(moll_array))
    moll_array = moll_array/(np.max(moll_array))*255.0
    return moll_array
#######

# changing data
t_start = time.time()
for num, path_ in enumerate(all_images):
    if not os.path.exists(destinationPath_data+'msim_%04i_data.tif'%num) or overwrite:
        image_data = hp.read_map(path_)
        image_data = np.array(image_data, np.float32)
        moll_array = hp.cartview(image_data, title=None, xsize=x_size, ysize=y_size, return_projected_map=True)
        if mod:
            moll_array = modification(moll_array)
        if norm:
            moll_array = normalization(moll_array)
        if rgb:
            moll_array = toRGB(moll_array)
        moll_image = Image.fromarray(moll_array)
        print('Saving to '+destinationPath_data+'msim_%04i_data.tif'%num)
        moll_image.save(destinationPath_data+'msim_%04i_data.tif'%num)

print('Total elapsed time:',time.time()-t_start,'s')
print('Done.')
