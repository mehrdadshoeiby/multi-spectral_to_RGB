"""
initialise the package and generate a dataloader class using args as 
the input
"""


from __future__ import print_function, division
import os
import torch



from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data import stereo_msi, mytransforms

class StereoMSIDatasetLoader(object):
    """
    all the training data should be stored in the same folder
    format for lr image  ==> "image_{}_lr2".format(idx)
    """

    def __init__(self, args):
        # training dataset parameters
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle
        self.crop_size = args.crop_size
        # load training data
        self.mytransform1 = transforms.Compose(
                        [mytransforms.RandomCrop(self.crop_size),
                         mytransforms.RandomRotate([0,90,180,270]),
                         mytransforms.RandomHorizontalFlip(0.5),
                         mytransforms.Normalize01(),                         
                         mytransforms.ToTensor()]
                         )
        # load testing data
        self.mytransform2 = transforms.Compose(
                        [mytransforms.Normalize01(),
                         mytransforms.ToTensor()]
                         )
        self.transformed_dataset = stereo_msi.StereoMSITrainDataset(args,
                                                    self.mytransform1)
        self.train_loader = DataLoader(self.transformed_dataset,
                                       self.batch_size,
                                       self.shuffle,
                                       num_workers=self.num_workers,
                                       worker_init_fn=np.random.seed(1))
#        def _init_fn(worker_id):
#            np.random.seed(12 + worker_id)
#        mytransform2 = mytransforms.ToTensor()
        # LOAD test data
        self.testing_dataset = stereo_msi.StereoMSIValidDataset(args,
                                                    self.mytransform2)
        self.test_loader = DataLoader(self.testing_dataset)

#  to get results on testing dataset
        self.results_dataset = stereo_msi.StereoMSITestDataset(args,
                                                    self.mytransform2)
        self.results_loader = DataLoader(self.results_dataset)
