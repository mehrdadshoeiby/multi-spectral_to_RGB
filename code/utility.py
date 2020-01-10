import os
import math
import time
import datetime
from functools import reduce

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc
from scipy import signal, interpolate

import hdf5storage
import cv2

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.init as init
import torch.nn as nn

from tools.utils import save_matv73

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args, loader):
        self.args = args
        self.ok = True
        self.log = torch.Tensor() # to store psnr for validation 
        self.log_accuracy = torch.Tensor() # to store validation loss based on total training loss.
        self.loader = loader
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if self.args.test_only==True:
                self.dir = '../experiment/' + args.save
            else:
                self.dir = '../experiment/' + now + "_" + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            str_transform = loader.mytransform1
            f.write("Data transforms:{}".format(str_transform))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
#        self.plot_accuracy(epoch)
        # uncomment when trainer.test() is written.
        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])
        self.log_accuracy = torch.cat([self.log_accuracy, log])

    def write_log(self, log, refresh=False):
        print(log)
        open(self.dir + '/log.txt', 'a') # my line
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch): 
        axis = np.linspace(1, epoch, epoch/self.args.save_every)
        label = 'SR on {}'.format(self.args.data_test)
        plt.ioff()
        fig = plt.figure()
        plt.title(label)
        plt.plot(
            axis,
            self.log[:].numpy(),
            label='Scale 2'
        )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/PSNR_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def plot_accuracy(self, epoch): 
        axis = np.linspace(1, epoch, epoch/self.args.save_every)
        label = 'SR on {}'.format(self.args.data_test)
        plt.ioff()
        fig = plt.figure()
        plt.title(label)
        plt.plot(
            axis,
            10*np.log10(self.log_accuracy[:].numpy()),
            label='Scale 2'
        )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig('{}/accuracy_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('sr', 'lr', 'hr')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(65535 / self.args.pixel_range)
            ndarr = normalized.permute(1, 2, 0).cpu().numpy()
            save_matv73('{}{}.mat'.format(filename, p), 'data', ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    
def calc_mrae(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def postprocess(img_res):
    img_res = torch.clamp(img_res*65535, max=65535,min=0)
    img_res = torch.round(img_res)
    img_res = np.squeeze(np.transpose(torch.Tensor.cpu(img_res).detach().numpy(),[3,2,1,0]),axis=3)
    return img_res

def self_ensemble(model,input_data):
    input_data1 = input_data
    input_data2 = np.flip(input_data,2)
    
    input_data3 = np.rot90(input_data1, k=1, axes=(2, 1))  
    input_data4 = np.rot90(input_data1, k=2, axes=(2, 1))
    input_data5 = np.rot90(input_data1, k=3, axes=(2, 1))
    
    input_data6 = np.rot90(input_data2, k=1, axes=(2, 1))
    input_data7 = np.rot90(input_data2, k=2, axes=(2, 1))
    input_data8 = np.rot90(input_data2, k=3, axes=(2, 1))
    
    input_data1 = np.expand_dims(input_data1, axis=0).copy()
    input_data2 = np.expand_dims(input_data2, axis=0).copy()
    input_data3 = np.expand_dims(input_data3, axis=0).copy()
    input_data4 = np.expand_dims(input_data4, axis=0).copy()
    input_data5 = np.expand_dims(input_data5, axis=0).copy()
    input_data6 = np.expand_dims(input_data6, axis=0).copy()
    input_data7 = np.expand_dims(input_data7, axis=0).copy()
    input_data8 = np.expand_dims(input_data8, axis=0).copy()
   
    input_data1 = torch.from_numpy(input_data1).float().cuda()
    input_data2 = torch.from_numpy(input_data2).float().cuda()
    input_data3 = torch.from_numpy(input_data3).float().cuda()
    input_data4 = torch.from_numpy(input_data4).float().cuda()
    input_data5 = torch.from_numpy(input_data5).float().cuda()
    input_data6 = torch.from_numpy(input_data6).float().cuda()
    input_data7 = torch.from_numpy(input_data7).float().cuda()
    input_data8 = torch.from_numpy(input_data8).float().cuda()
    
    img_res1 = model(input_data1)
    img_res1 = postprocess(img_res1)

    img_res2 = model(input_data2)
    img_res2 = postprocess(img_res2)
    img_res2 = np.flip(img_res2,0)
    
    img_res3 = model(input_data3)
    img_res3 = postprocess(img_res3)
    img_res3 = np.rot90(img_res3, k=3, axes=(0, 1))
    
    img_res4 = model(input_data4)
    img_res4 = postprocess(img_res4)
    img_res4 = np.rot90(img_res4, k=2, axes=(0, 1))
    
    img_res5 = model(input_data5)
    img_res5 = postprocess(img_res5)
    img_res5 = np.rot90(img_res5, k=1, axes=(0, 1))
    
    img_res6 = model(input_data6)
    img_res6 = postprocess(img_res6)
    img_res6 = np.flip(img_res6,0)
    img_res6 = np.rot90(img_res6, k=1, axes=(0, 1))                       
    
    img_res7 = model(input_data7)
    img_res7 = postprocess(img_res7)
    img_res7 = np.flip(img_res7,0)
    img_res7 = np.rot90(img_res7, k=2, axes=(0, 1)) 
    
    img_res8 = model(input_data8)
    img_res8 = postprocess(img_res8)
    img_res8 = np.flip(img_res8,0)
    img_res8 = np.rot90(img_res8, k=3, axes=(0, 1)) 
    return np.round((img_res1+img_res2+img_res3+img_res4+img_res5+
                     img_res6+img_res7+img_res8)/8)
