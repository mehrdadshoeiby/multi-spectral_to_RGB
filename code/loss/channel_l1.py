import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

class ChannelL1(nn.Module):
    def __init__(self):
        super(ChannelL1, self).__init__()
        
    def forward(self, sr, hr):
        sr_red = sr.narrow(1,0,1) # channel red, green, or blue?
        sr_green = sr.narrow(1,1,1) # channel red, green, or blue?
        sr_blue = sr.narrow(1,2,1) # channel red, green, or blue?
        hr_red = hr.narrow(1,0,1) # channel red, green, or blue?
        hr_green = hr.narrow(1,1,1) # channel red, green, or blue?
        hr_blue = hr.narrow(1,2,1) # channel red, green, or blue?        
        l1_red = F.smooth_l1_loss(sr_red, hr_red)
        l1_blue = F.smooth_l1_loss(sr_blue, hr_blue)
        l1_green = F.smooth_l1_loss(sr_green, hr_green)
        l1_mean = (2*l1_red + 2*l1_blue + l1_green)/3
        
        return l1_mean
        