import os
from importlib import import_module

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# every loss function is the child of the _Loss class
class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')
        self.args = args
        self.ckp = ckp
        self.loss = []
        self.data_length = ckp.loader.train_loader.__len__()
        self.epoch = 0
        self.bp_counter = 0 #back prop counter
        # nn.ModuleList() stores nn.Modules (here different types of 
        # loss) Just like a Python list
        self.loss_module = nn.ModuleList()
        # ... on the go. This needs to be modified depending on the loss.
        # w1*l1 + w2*l2 + w3*l3 + ...
	   # the code account for if different losses are being used and
        # uses all of them
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss().cuda()
            elif loss_type == 'SmoothL1':
                loss_function = nn.SmoothL1Loss().cuda()
            elif loss_type == 'ChannelL1':
                module = import_module('loss.channel_l1')
                loss_function = module.ChannelL1().cuda()
            elif loss_type == 'SID':
                module = import_module('loss.sid')
                loss_function = module.SID().cuda()
            elif loss_type == 'MRAE':
                module = import_module('loss.mrae')
                loss_function = module.MRAE().cuda()
            elif loss_type == 'tv':
                module = import_module('loss.tv')
                loss_function = module.TV().cuda()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

#       adverserial code goes here            
        # loss is a list, append the following is more than 1 loss is
        # being used (the last elemnt of loss)
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        # loss_module is not used in the actual implementation! 
        # Is it only for adverserial?
        for l in self.loss: 
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                # store the actual loss-fucntion in loss_module (list)
                self.loss_module.append(l['function'])
        # this log is to store training loss, dont confuse with the other log
        # also there is a self.log in checkpoint to store validation/testing loss.
        self.log = torch.Tensor()
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device) # do I have to move loss to device too?
        # if more than one GPU
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )
        # if a directory is givn, load the checkpoints.
        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)
    # sums up all the different type of losses, this is done for all 
    # the batches, so end_log devides log[-1] by number of the batches
    def forward(self, sr, sr_, hr):
        self.bp_counter = self.bp_counter + 1
        self.epoch = self.bp_counter/self.data_length
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:                
                if l['type']=='tv':
                    loss = l['function'](sr) + l['function'](sr_)
                    effective_loss = l['weight'] * loss * np.exp(-self.epoch/self.args.tau)
                else:
                    loss = l['function'](sr, hr) + l['function'](sr_, hr)
                    effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # the last element
                self.log[-1, i] += effective_loss.item()
#            elif l['type'] == 'DIS': # used for GAN
#                self.log[-1, i] += self.loss[i - 1]['function'].loss
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        return loss_sum
    # adverserial loss has a scheduler attribute 
    # (not used for normal loss funcitons)
#    def step(self):
#        for l in self.loss_module():
#            if hasattr(l, 'scheduler'):
#                l.scheduler.step()
    def valid_loss(self, sr, sr_, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:            
                if l['type']=='tv':
                    loss = l['function'](sr) + l['function'](sr_)
                    effective_loss = l['weight'] * loss * np.exp(-self.epoch/self.args.tau)
                else:
                    loss = l['function'](sr, hr) + l['function'](sr_, hr)
                    effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # the last element
#            elif l['type'] == 'DIS': # used for GAN
#                self.log[-1, i] += self.loss[i - 1]['function'].loss
        loss_sum = sum(losses)
        return loss_sum
        
                
    def start_log(self): # builds a zero tensor with the dimention of loss
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches): # end the log of loss
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))
        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = 'log10({} Loss)'.format(l['type'])
            plt.ioff()
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, 10*np.log10(self.log[:, i].numpy()), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            if (epoch % self.args.save_every==0 and
                (l['type'] == 'Total' or len(self.loss)==1)):
                axis = np.linspace(self.args.save_every, epoch, epoch/self.args.save_every)
                plt.plot(
                    axis,
                    10*np.log10(self.ckp.log_accuracy[:].numpy()),
                    label='Accuracy'
                )
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module: # only used for adverserial
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

