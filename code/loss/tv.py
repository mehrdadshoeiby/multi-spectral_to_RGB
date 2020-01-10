# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def roll(x, n, dim=2):
    if dim==0:
        return torch.cat((x[-n:], x[:-n]))
    if dim==1:
        return torch.cat((x[:,-n:], x[:,:-n]), dim=1)
    if dim==2:
        return torch.cat((x[:,:,-n:], x[:,:,:-n]), dim=2)
    if dim==3:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n]), dim=3)
        
class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()
        
    def forward(self, sr):
        # gradient of primal variable
        grad_sr_x = roll(sr,-1,dim=2)-sr # x-component of U's gradient
        grad_sr_y = roll(sr,-1,dim=3)-sr # y-component of U's gradient        
        grad_sr = grad_sr_x + grad_sr_y
        tv = torch.abs(grad_sr)
        
        return torch.mean(tv)
