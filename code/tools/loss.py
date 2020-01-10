#!/usr/local/bin/python
from __future__ import division
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class reconstruct_loss(nn.Module):
    """the loss between the input and synthesized input"""
    def __init__(self, cie_matrix, batchsize):
        super(reconstruct_loss, self).__init__()
        self.cie = Variable(torch.from_numpy(cie_matrix).float().cuda(), requires_grad=False)
        self.batchsize = batchsize
    def forward(self, network_input, network_output):
        network_output = network_output.permute(3, 2, 0, 1)
        network_output = network_output.contiguous().view(-1, 31)
        reconsturct_input = torch.mm(network_output,self.cie)
        reconsturct_input = reconsturct_input.view(50, 50, 64, 3)
        reconsturct_input = reconsturct_input.permute(2,3,1,0)
        reconstruction_loss = torch.mean(torch.abs(reconsturct_input - network_input))                  
        return reconstruction_loss

def rrmse_loss(outputs, label):
    """Computes the rrmse value"""
    error = torch.abs(outputs-label)/(label + 1/65535)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def rrmse_loss2(outputs, label):
    """Computes the rrmse value"""
    zeros = torch.zeros(outputs.shape)
    outputs = torch.where(outputs>1/65535,outputs,zeros.cuda())
    error = torch.abs(outputs-label)/(label + 1/65535)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def rrmse_loss3(outputs, label):
    """Computes the rrmse value"""
    zeros = torch.zeros(outputs.shape)
    ones = torch.ones(outputs.shape)*1/65535
    twos = torch.ones(outputs.shape)*2/65535
    outputs = torch.where(outputs>1/65535,outputs,zeros.cuda())
    outputs = torch.where((outputs>2/65535)|(outputs<1/65535),outputs,ones.cuda())
    outputs = torch.where((outputs>3/65535)|(outputs<2/65535),outputs,twos.cuda())

    error = torch.abs(outputs-label)/(label + 1/65535)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def rrmse_loss_round(outputs, label):
    """Computes the rrmse value"""
    outputs = torch.clamp(outputs*65535,max=65535,min=0)
    outputs = torch.round(outputs)
    error = torch.abs(outputs-label)/(label + 1)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def rrmse_loss_ceil(outputs, label):
    """Computes the rrmse value"""
    outputs = torch.clamp(outputs*65535,max=65535,min=0)
    outputs = torch.ceil(outputs)-1
    error = torch.abs(outputs-label)/(label + 1)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def rrmse_loss_con(outputs, label):
    """Computes the rrmse value"""
    zeros = torch.zeros(outputs.shape)
    outputs = torch.clamp(outputs*65535,max=65535,min=0)
    outputs = torch.round(torch.where(outputs>1,outputs,zeros.cuda()))
    error = torch.abs(outputs-label)/(label + 1)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def sid_loss(output, target):
    """For the network, the input image dimension is BxCxHxW"""
    a = torch.sum(torch.sum(output* torch.log10(torch.clamp((output + 1e-3/65536)/(target + 1e-3/65536),min=1e-8)),3),2)
    b = torch.sum(torch.sum(target* torch.log10(torch.clamp((target+ 1e-3/65536)/(output+ 1e-3/65536),min=1e-8)),3),2)
    sid = torch.sum(torch.abs(a + b))/(target.shape[0]*target.shape[1]*target.shape[2] * target.shape[3])
    return sid 

def test_sid_loss(output, target):
    """For the network, the input image dimension is BxCxHxW"""
    a = torch.sum(torch.sum(output* torch.log10(torch.clamp((output + 1e-3)/(target + 1e-3),min=1e-8)),3),2)
    b = torch.sum(torch.sum(target* torch.log10(torch.clamp((target+ 1e-3)/(output+ 1e-3),min=1e-8)),3),2)
    sid = torch.sum(torch.abs(a + b))/(target.shape[0]*target.shape[1]*target.shape[2] * target.shape[3])
    return sid 

def appsa_loss(output, target):
    nom = torch.sum(target* output, dim=1)
    denom = torch.norm(target,2,1) * torch.norm(output,2,1) 
    cos = torch.clamp(nom/(denom + 1e-3/65536), max=1)
    appsa = torch.acos(torch.clamp(cos,min=1e-8))
    appsa = torch.sum(appsa.view(-1))/(target.shape[0]*target.shape[2] * target.shape[3])
    return appsa

def test_appsa_loss(output, target):
    nom = torch.sum(target* output, dim=1)
    denom = torch.norm(target,2,1) * torch.norm(output,2,1) 
    cos = torch.clamp(nom/(denom + 1e-3), max=1)
    appsa = torch.acos(torch.clamp(cos,min=1e-8))
    appsa = torch.sum(appsa.view(-1))/(target.shape[0]*target.shape[2] * target.shape[3])
    return appsa



def tvloss(output, tv_weight):
    """Computes the total variation loss"""
    diff_i = torch.sum(torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]))    
    tv_loss = tv_weight*(diff_i + diff_j)
    return tv_loss

def rrmse_loss_round(outputs, label):
    """Computes the rrmse value"""
    outputs = torch.clamp(outputs*65535,max=65535,min=0)
    outputs = torch.round(outputs)
    error = torch.abs(outputs-label)/(label + 1)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def rrmse_loss_ceil(outputs, label):
    """Computes the rrmse value"""
    outputs = torch.clamp(outputs*65535,max=65535,min=0)
    outputs = torch.ceil(outputs)-1
    error = torch.abs(outputs-label)/(label + 1)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def rrmse_loss_con(outputs, label):
    """Computes the rrmse value"""
    zeros = torch.zeros(outputs.shape)
    outputs = torch.clamp(outputs*65535,max=65535,min=0)
    outputs = torch.round(torch.where(outputs>1,outputs,zeros.cuda()))
    error = torch.abs(outputs-label)/(label + 1)   #1/65536 = 1.5e-5
    rrmse = torch.mean(error.view(-1))
    return rrmse

def sid_loss(output, target):
    """For the network, the input image dimension is BxCxHxW"""
    a = torch.sum(torch.sum(output* torch.log10(torch.clamp((output + 1e-3/65536)/(target + 1e-3/65536),min=1e-8)),3),2)
    b = torch.sum(torch.sum(target* torch.log10(torch.clamp((target+ 1e-3/65536)/(output+ 1e-3/65536),min=1e-8)),3),2)
    sid = torch.sum(torch.abs(a + b))/(target.shape[0]*target.shape[1]*target.shape[2] * target.shape[3])
    return sid 

def test_sid_loss(output, target):
    """For the network, the input image dimension is BxCxHxW"""
    a = torch.sum(torch.sum(output* torch.log10(torch.clamp((output + 1e-3)/(target + 1e-3),min=1e-8)),3),2)
    b = torch.sum(torch.sum(target* torch.log10(torch.clamp((target+ 1e-3)/(output+ 1e-3),min=1e-8)),3),2)
    sid = torch.sum(torch.abs(a + b))/(target.shape[0]*target.shape[1]*target.shape[2] * target.shape[3])
    return sid 

def appsa_loss(output, target):
    nom = torch.sum(target* output, dim=1)
    denom = torch.norm(target,2,1) * torch.norm(output,2,1) 
    cos = torch.clamp(nom/(denom + 1e-3/65536), max=1)
    appsa = torch.acos(torch.clamp(cos,min=1e-8))
    appsa = torch.sum(appsa.view(-1))/(target.shape[0]*target.shape[2] * target.shape[3])
    return appsa

def test_appsa_loss(output, target):
    nom = torch.sum(target* output, dim=1)
    denom = torch.norm(target,2,1) * torch.norm(output,2,1) 
    cos = torch.clamp(nom/(denom + 1e-3), max=1)
    appsa = torch.acos(torch.clamp(cos,min=1e-8))
    appsa = torch.sum(appsa.view(-1))/(target.shape[0]*target.shape[2] * target.shape[3])
    return appsa



def tvloss(output, tv_weight):
    """Computes the total variation loss"""
    diff_i = torch.sum(torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]))    
    tv_loss = tv_weight*(diff_i + diff_j)
    return tv_loss
