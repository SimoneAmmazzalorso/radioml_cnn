#!/usr/bin/env python
#import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
#import matplotlib.pyplot as plt

import glob
import os
import time
from PIL import Image
#from multiprocessing import Pool

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action='store_true', help="overwrite maps.")
parser.add_argument("--xsize_old", type=int, help="xsize of the original tif figure.", default=2000)
parser.add_argument("--xsize_new", type=int, help="xsize of the new tif figure.", default=200)
parser.add_argument("--fix_x", type=int, help="fix x pixel.", default=-1)
parser.add_argument("--fix_y", type=int, help="fix y pixel.", default=-1)
parser.add_argument("--path", type=str, help="path of source files.", default='')
parser.add_argument("--path_dest", type=str, help="path of destination files.", default='')
parser.add_argument("--tag", type=str, help="tag of destination files.", default='')

args = parser.parse_args()
overwrite = args.overwrite
#x_size_old = args.xsize_old
x_size_new = args.xsize_new
fix_x = args.fix_x
fix_y = args.fix_y
path = args.path
destinationPath_data = args.path_dest
tag = args.tag

if tag is not '':
   tag = tag+'_'

# transform all .fits maps in dircetory into .tif

all_images = glob.glob(os.path.join(path, "*.tif")) # changed output name for easier identification

print('There are %i maps in %r'%(len(all_images), path))
image = np.array(Image.open(all_images[0]))

x_size_old = image.shape[1]    # the shorter side
y_size_old = image.shape[0]
print('...of size:',x_size_old,'X',y_size_old)

#y_size_old = int(x_size_old/2)
y_size_new = int(x_size_new/2)
print('Producing tif maps of size:',x_size_new,'X',y_size_new)

x_index_list = np.arange(0,x_size_old,step=x_size_new)
y_index_list = np.arange(0,y_size_old,step=y_size_new)
t_start = time.time()
for num, path_ in enumerate(all_images):
    if not os.path.exists(destinationPath_data+'msim_'+tag+'%04i_data.tif'%num) or overwrite:
        image_data = np.array(Image.open(path_))
        if fix_x != -1 and fix_x > 0 and fix_x <= x_index_list[-1]:
            x_index = fix_x
        else:
            x_index = np.random.choice(x_index_list,1)[0]
        if fix_y != -1 and fix_y > 0 and fix_y <= y_index_list[-1]:
            y_index = fix_y
        else:
            y_index = np.random.choice(y_index_list,1)[0]
        print('pixel coordinates:',x_index,y_index)
        new_image = image_data[y_index:y_index+y_size_new,x_index:x_index+x_size_new]
        #print(new_image.shape)
        #plt.imshow(new_image)
        #plt.show()
        #exit()
        new_image = Image.fromarray(new_image)
        print('Saving to '+destinationPath_data+'msim_'+tag+'%04i_data.tif'%num)
        new_image.save(destinationPath_data+'msim_'+tag+'%04i_data.tif'%num)

print('Total elapsed time:',time.time()-t_start,'s')
print('Done.')
