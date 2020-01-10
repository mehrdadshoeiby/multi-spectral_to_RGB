# -*- coding: utf-8 -*-
"""
define data loader class
"""

from __future__ import print_function, division
import os
import glob
import torch
import pandas as pd
from skimage import transform
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

#import spectral.io.envi as envi

from data.tools import read_mat
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
    
class StereoMSITrainDataset(Dataset):
    """
    all the training data should be stored in the same folder
    format for lr image  ==> "image_{}_lr2".format(idx)
    """
    
    def __init__(self, args, mytransform):
        self.root_dir = args.root_dir
        self.mytransform = mytransform
 
    def __len__(self):
        # find the number of labels hence lenght of the dataset.
        return len(glob.glob1(os.path.join(self.root_dir + 'data/train'),
                              '*.tiff'))

    def __getitem__(self, idx):
        # dataset idx starts from 1
        im_lr = os.path.join(self.root_dir,
                                "data/train/{}.mat".format(idx+1))
        im_hr = os.path.join(self.root_dir,
                                "data/train/{}.tiff".format(idx+1))

        im_lr = np.array(read_mat(im_lr), dtype=float)
        im_hr = np.array(imread(im_hr), dtype=float)
        sample_train = {'im_lr': im_lr, 'im_hr': im_hr}

        if self.mytransform:
            sample_train = self.mytransform(sample_train)
        return sample_train

class StereoMSIValidDataset(Dataset):
    """
    all the training data should be stored in the same folder
    format for lr image  ==> "image_{}_lr2".format(idx)
    """
    
    def __init__(self, args, mytransform):
        self.root_dir = args.root_dir
        self.mytransform = mytransform
 
    def __len__(self):
        # find the number of labels hence lenght of the dataset.
        return len(glob.glob1(os.path.join(self.root_dir + 'data/valid'),
                              '*.tiff'))

    def __getitem__(self, idx):
        # validation dataset idx starts from 201
        im_lr = os.path.join(self.root_dir,
                                "data/valid/{}.mat".format(str(idx+251)))
        im_hr = os.path.join(self.root_dir,
                                "data/valid/{}.tiff".format(str(idx+251)))

        im_lr = np.array(read_mat(im_lr), dtype=float)
        im_hr = np.array(imread(im_hr), dtype=float)
        sample_valid = {'im_lr': im_lr, 'im_hr': im_hr}

        if self.mytransform:
            sample_valid = self.mytransform(sample_valid)
        # read validation/testing dataset

        return sample_valid

class StereoMSITestDataset(Dataset):
    """
    all the training data should be stored in the same folder
    format for lr image  ==> "image_{}_lr2".format(idx)
    """
    
    def __init__(self, args, mytransform):
        self.root_dir = args.root_dir
        self.mytransform = mytransform
 
    def __len__(self):
        # find the number of labels hence lenght of the dataset.
        return len(glob.glob1(os.path.join(self.root_dir + 'data/test'),
                              '*.tiff'))

    def __getitem__(self, idx):
        # validation dataset idx starts from 201
        im_lr = os.path.join(self.root_dir,
                                "data/test/{}.mat".format(str(idx+276)))
        im_hr = os.path.join(self.root_dir,
                                "data/test/{}.tiff".format(str(idx+276)))

        im_lr = np.array(read_mat(im_lr), dtype=float)
        im_hr = np.array(imread(im_hr), dtype=float)
        sample_valid = {'im_lr': im_lr, 'im_hr': im_hr}

        if self.mytransform:
            sample_valid = self.mytransform(sample_valid)
        # read validation/testing dataset

        return sample_valid
