# -*- coding: utf-8 -*-
"""
generate training validation, and testing dataset for
 multispectral to RGB conversion.
"""


from shutil import copyfile # copyfile(src, dst)
import sys
sys.path.insert(0, '/mnt/md0/CSIRO/projects/XiQ_cameras/')
import os
from random import choice
import XiQ_class

from skimage.io import imread, imsave
import numpy as np
import time

import hdf5storage
import h5py

def read_mat(im, varname='data'):
    f = h5py.File(im,'r') 
    im = f.get(varname) 
    im = np.array(im, float).T
    
    return im
def write_to_list(my_list, file_path):    
    outF = open(file_path, "w")
    for line in my_list:
        # write line to output file
        outF.write(line)
        outF.write("\n")
    outF.close()

im_dir_ms = (r"/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/"
          r"main/data_raw/multispectral/")
im_dir_rgb = (r"/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/"
          r"main/data_raw/regist_pwc-net/rgb_registered/")

output_dir = (r"/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/"
              r"main/RCAN_v7/data/")

#bad_images = ["0009.tif", "0014.tif", "0015.tif", "0018.tif", "0023.tif",
#              "0027.tif", "0028.tif", "0031.tif", "0033.tif", "0035.tif",
#              "0043.tif", "0048.tif", "0054.tif", "0050.tif", "0071.tif",
#              "0073.tif", "0078.tif", "0079.tif", "0082.tif", "0085.tif",
#              "0089.tif", "0091.tif", "0092.tif", "0097.tif", "0098.tif", 
#              "0099.tif", "0100.tif", "0103.tif", "0105.tif", "0118.tif",
#              "0120.tif", "0121.tif", "0122.tif", "0124.tif", "0125.tif", 
#              "0130.tif", "0131.tif", "0135.tif", "0143.tif", "0147.tif",
#              "0166.tif", "0172.tif", "0173.tif", "0206.tif", "0221.tif",
#              "0253.tif", "0267.tif", "0310.tif", "0326.tif",  "0342.tif",
#              "0348.tif"]
# 
#train_images = []
#valid_images = []
#test_images = []
#image_list = list(set(os.listdir(im_dir_ms))-set(bad_images))
#image_list_temp = image_list.copy()
dx = 112*4
dy = 56*4

#while len(train_images)<250:
#    # generate training
#    name = choice(image_list_temp)
#    train_images.append(name)
#    image_list_temp.remove(name)

#write_to_list(train_images, output_dir + 'train_images.txt')
train_dir = output_dir + "train/"
os.system("rm -rf " + train_dir)
os.mkdir(train_dir)
for i, name in enumerate(train_images, 1):
    print("{}th training image".format(i))
    camera = XiQ_class.XiQ_470to620nm(im_dir_ms,name)
    camera.image_savemat_cropped(train_dir, str(i), dx, dy, interp=False,
                                 mode='same', normalize=False)
    time.sleep(1)
    im_hr = imread(im_dir_rgb + 'forward_' + name +'f')
    im_hr = im_hr[dy:-dy,dx:-dx,:]
    im_hr = np.uint8((im_hr / im_hr.max()) * 255)
    imsave(train_dir + "{}.tiff".format(i), im_hr)

#######################################################################
    #generate validation
#image_list1 = list(set(image_list)-set(train_images))
#image_list_temp1 = image_list1.copy()
#while len(valid_images)<25:
#    name = choice(image_list_temp1)
#    valid_images.append(name)
#    image_list_temp1.remove(name)

#write_to_list(valid_images, output_dir + 'valid_images.txt')
valid_dir = output_dir + "valid/"
os.system("rm -rf " + valid_dir)
os.mkdir(valid_dir)

for i, name in enumerate(valid_images, 1):
    print("{}th validation image".format(i))
    camera = XiQ_class.XiQ_470to620nm(im_dir_ms,name)
    camera.image_savemat_cropped(valid_dir, str(i+250), dx, dy, interp=False,
                                 mode='same', normalize=False)    
    time.sleep(1) 
    im_hr = imread(im_dir_rgb + 'forward_' + name +'f')
    im_hr = im_hr[dy:-dy,dx:-dx,:]
    imsave(valid_dir + "{}.tiff".format(i+250), im_hr)
#######################################################################
    #generate testing
    # choose the rest of the images as testing.
    # if the images name is in bad_images remove do nothing
    # else append to an empty list of images. 
#image_list2 = list(set(image_list1)-set(valid_images))
#image_list_temp2 = image_list2.copy()
#while len(test_images)<25:
#    name = choice(image_list_temp2)
#    test_images.append(name)
#    image_list_temp2.remove(name)

#write_to_list(test_images, output_dir + 'test_images.txt')
test_dir = output_dir + "test/"
os.system("rm -rf " + test_dir)
os.mkdir(test_dir)

for i, name in enumerate(test_images, 1):
    print("{}th testing image".format(i))
    camera = XiQ_class.XiQ_470to620nm(im_dir_ms,name)
    camera.image_savemat_cropped(test_dir, str(i+275), dx, dy, interp=False,
                                 mode='same', normalize=False)    
    time.sleep(1)
    im_hr = imread(im_dir_rgb + 'forward_' + name +'f')
    im_hr = im_hr[dy:-dy,dx:-dx,:]
    imsave(test_dir + "{}.tiff".format(i+275), im_hr)
