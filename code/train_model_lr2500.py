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

import random

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
args.loss = '1*ChannelL1' # L1, SmoothL1, ChannelL1, SID, MSE, MRAE, LocalSmoothL1, TV
args.tau = 500
# dataset properties
args.root_dir = "/flush3/sho092/RCAN_v7/"
#args.root_dir = (r"/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/"
#                 r"main/RCAN_v7/")
#args.data_dir = ("/home/sho092/Documents/CSIRO/my_data/RCAN_v7/")

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
args.epochs = 15000
args.optimizer = 'ADAM'
args.beta1 = 0.9
args.beta2 = 0.999
args.epsilon = 1e-8
# args.momentum = 0.9
args.lr = 1e-4
args.weight_decay = 0
# scheduler parameters
args.decay_type = 'step'
args.lr_decay = 2500
args.gamma = 0.5
args.reset = True # works when the saved directory doesnt change. 
# log specification
args.print_model = True
args.pre_train = '.'#args.dir + 'experiment/2019-02-26-10:53:50_1*ChannelL1+1*tv/model/model_latest.pt'
args.resume = 0
args.load = '.'#'2019-02-22-14:26:29_1*ChannelL1+1*tv'
args.cpu = False # 'store_true'
args.n_GPUs = 2
args.test_only = False
# saving and loading models
args.save_every = 50 
args.save_models = True # saves all intermediate models
# file name to saspyder2ve, if '.' the name is date+time
args.save = args.loss
args.save_results = True
loader = dataloader.StereoMSIDatasetLoader(args)
checkpoint = utility.checkpoint(args, loader)
debug_dir = os.path.join(checkpoint.dir, "debug_results")
os.mkdir(debug_dir)
my_loss = loss.Loss(args, checkpoint) if not args.test_only else None
my_model = network.Model(args, checkpoint)
my_model.apply(weight_init)
t = Trainer(args, loader, my_model, my_loss, checkpoint)
i=True
#args.test_only = True
#t.test_model()
#if args.test_only==True:
#    t.test_model()
while not t.terminate():
    t.train()
    # train model
    if t.epoch() % args.save_every==0:
        t.test()
        if (t.epoch() % 50==0):
            imsave(debug_dir + "/sr_epoch_{}.png".format(t.epoch()),
                   normalise01(np.float64(t.sr_valid)))
            if i:
                imsave(debug_dir + "/lr.png",
                       normalise01(np.float64(t.lr_valid)))
                imsave(debug_dir + "/hr.png",
                       normalise01(np.float64(t.hr_valid)))
                i=False                     
checkpoint.done()
