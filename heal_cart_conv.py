#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
#import matplotlib.pyplot as plt

import glob
import os
import time
from PIL import Image
import healpy as hp
#from multiprocessing import Pool

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action='store_true', help="overwrite maps.")
parser.add_argument("--xsize", type=int, help="xsize of the output tif figure.", default=2000)
parser.add_argument("--path", type=str, help="path of source files.", default='')
parser.add_argument("--path_dest", type=str, help="path of destination files.", default='')
parser.add_argument("--tag", type=str, help="tag of destination files.", default='')

args = parser.parse_args()
overwrite = args.overwrite
x_size = args.xsize
path = args.path
destinationPath_data = args.path_dest
tag = args.tag

if tag is not '':
   tag = tag+'_'

# transform all .fits maps in dircetory into .tif

all_files = glob.glob(os.path.join(path, "*.fits")) # changed output name for easier identification

print('There are %i maps in %r'%(len(all_files), path))

y_size = int(x_size/2)
print('Producing tif maps of size:',x_size,'X',y_size)

t_start = time.time()
for num, path_ in enumerate(all_files):
    if not os.path.exists(destinationPath_data+'msim_'+tag+'%04i_data.tif'%num) or overwrite:
        image_data = hp.read_map(path_)
        new_image = hp.cartview(image_data, title=None, xsize=x_size, ysize=y_size, return_projected_map=True)
        #print(new_image.shape)
        #plt.show()
        #exit()
        new_image = Image.fromarray(new_image)
        print('Saving to '+destinationPath_data+'msim_'+tag+'%04i_data.tif'%num)
        new_image.save(destinationPath_data+'msim_'+tag+'%04i_data.tif'%num)

print('Total elapsed time:',time.time()-t_start,'s')
print('Done.')
