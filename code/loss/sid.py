import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable



class SID(nn.modules.loss._Loss):
    def __init__(self):
        super(SID, self).__init__()

    def forward(self, sr, hr):      
        for i in range(14):
            hr1 = torch.zeros(hr.shape).cuda()
            sr1 = torch.zeros(sr.shape).cuda()
            hr1[:,i,:,:] = hr[:,i,:,:] / torch.sum(hr[:i,:,:])
            sr1[:,i,:,:] = sr[:,i,:,:] / torch.sum(sr[:,i,:,:])
        N = hr1.shape[1]
        err = torch.zeros(N).cuda()
        for i in range(N):
            err[i] = abs(torch.sum(sr1[:,i,:,:] * torch.log10(
                                 (sr1[:,i,:,:] + 1)/(hr1[:,i,:,:] + 1)
                                                     )) +
                         torch.sum(hr1[:,i,:,:] * torch.log10(
                                 (hr1[:,i,:,:] + 1)/(sr1[:,i,:,:] + 1)))
                                 )
        return torch.sum(err)
