#import spectral.io.envi as envi
import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import h5py

#def read_fla_file(filename):
#    fla_file = envi.open(filename + '.hdr', filename + '.fla')
#    im = fla_file.load(scale=False)
#    return im

def read_mat(im, varname='data'):
    f = h5py.File(im,'r') 
    im = f.get(varname) 
    im = np.array(im, float).T
    
    return im   
    
#def mean_std(args, im_lr_nam=None):
#    # calculate mean and std for training track1 PIRM2018 challenge
#    if im_lr_name==None:
#        im_lr_name = os.path.join(args.root_dir,"train/image_{}_lr2".format(1))
#        im_lrr = np.array(read_fla_file(im_lr_name), dtype=float)
#    for i in range(2,201,1):        
#        im_lr_name = os.path.join(args.root_dir,
#                                "train/image_{}_lr2".format(i))
#
#        im_lr = np.array(read_fla_file(im_lr_name), dtype=float)
#        im_lrr = np.vstack((im_lrr, im_lr))
#    im_lr_mean = im_lrr.mean(0).mean(0)
#    im_lr_mean = ndarray.tolist(im_lr_mean)
#    im_lr_std = [np.std(im_lrr[:,:,j]) for j in range(14)]    
#
#    return im_lr_mean, im_lr_std
    
#def display_valid(args):
#    """
#    display validation along with their SR
#    """
#    results_dir = (r"/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/"
#                   r"Train_PRIM_VIDAR_code/experiment/"
#                   r"2019-01-30-15:33:53_1*MSE_Normalized:False")
#    for i in np.arange(201, 221, 1):
#        im_lr_valid = os.path.join(args.root_dir,"valid/image_{}_lr2".format(i))
#        j = i-200
#        im_sr_valid = os.path.join(results_dir,"results/image_{}_sr_x2_sr.mat".format(j))
#        im_lr = read_fla_file(im_lr_valid)
#        im_sr = read_mat(im_sr_valid, 'data')
#        plt.figure()
#        plt.imshow(np.average(im_lr, axis=2))
#        plt.figure()        
#        plt.imshow(np.average(im_sr, axis=2))