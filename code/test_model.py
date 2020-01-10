# -*- coding: utf-8 -*-
"""
train VIDAR's model from PIRM2018
"""
        
#from __future__ import division
import torch
import torch.nn as nn
from torchvision import transforms, utils
from importlib import reload
import os
import numpy as np
from skimage.io import imsave, imread

import matplotlib.pyplot as plt

import loss
from trainer import Trainer
import network
#import network.net_track1 as model
from tools.utils import save_matv73, AverageMeter
#from loss import Loss
from data import stereo_msi, mytransforms, dataloader
import utility
from utility import weight_init

class args:
    def __init__(self):
        super(CALayer, self).__init__()
        # model args        
        args.n_resgroups 
        args.n_resblocks
        args.n_feats 
        args.n_reduction
        # loss args        
        args.loss_type
        # data args
        args.root_dir
        args.transform
    
def normalise01(a):
    """
    normalise a 2D array to [0, 1]
    """    
    b = (a - a.min()) / (a.max() - a.min())    
    return b


#def main():
# model properties
args.model = 'RCAN'
args.n_resgroups = 5
args.n_resblocks = 3
args.n_feats = 64
args.n_reduction = 16
args.scale = 2
args.self_ensemble = False
args.precision = 'single'
# loss properties
args.loss = '1*ChannelL1+1*tv' # L1, SmoothL1, ChannelL1, SID, MSE, MRAE, LocalSmoothL1, TV
args.tau = 500
# dataset properties
args.root_dir = "/flush3/sho092/RCAN_v7/"
#args.root_dir = (r"/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/"
#                 r"main/RCAN_v7/")
args.batch_size = 10
args.num_workers = 24
args.crop_size = 120
args.shuffle = True
args.pixel_range = 65535
args.data_test = "StereoMSI-valid"
args.normalize_data=True
#
args.skip_threshold = 1e6
args.print_every = 20
# optimiser parameters
args.epochs = 5000
args.optimizer = 'ADAM'
args.beta1 = 0.9
args.beta2 = 0.999
args.epsilon = 1e-8
# args.momentum = 0.9
args.lr = 1e-4
args.weight_decay = 0
# scheduler parameters
args.decay_type = 'step'
args.lr_decay = 1500
args.gamma = 0.5
args.reset = True # works when the saved directory doesnt change. 
#   log specification
args.print_model = True
args.resume = 0
args.resume = 0
args.load = '.'#'2019-02-22-14:26:29_1*ChannelL1+1*tv'
args.cpu = False # 'store_true'
args.n_GPUs = 2
args.test_only = True
# saving and loading models
args.save_every = 100
args.save_models = True # saves all intermediate models
folder = "2019-03-12-10:04:18_1*SmoothL1"
model = "8000"
args.pre_train = args.root_dir + 'experiment/{}/model/model_{}.pt'.format(folder, model)
# file name to save, if '.' the name is date+time
if args.test_only==False:
    args.save = args.loss
else: args.save = "{}_Testing".format(folder)
args.save_results = True
loader = dataloader.StereoMSIDatasetLoader(args)
checkpoint = utility.checkpoint(args, loader)    
# make directory if does not exist        
test_dir = os.path.join(checkpoint.dir, 'test_results')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
test_model_dir = os.path.join(test_dir, 'model_{}/'.format(model))
os.mkdir(test_model_dir)
my_loss = loss.Loss(args, checkpoint) if not args.test_only else None
my_model = network.Model(args, checkpoint)
t = Trainer(args, loader, my_model, my_loss, checkpoint)    
if args.test_only==True:
    psnr = t.test_model(test_model_dir)
my_file = open(args.root_dir + "experiment/" + args.save + "/metrics.txt",'a')
my_file.writelines("PSNR: {}".format(psnr))
my_file.close()
