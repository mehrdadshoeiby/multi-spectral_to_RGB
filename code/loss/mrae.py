import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable



class MRAE(nn.Module):
    def __init__(self):
        super(MRAE, self).__init__()
        
    def forward(self, output, target, mask=None):
        relative_diff = torch.abs(output - target) / (target + 1.0/65535.0)
        if mask is not None:
            relative_diff = mask * relative_diff
        return torch.mean(relative_diff)

#class SID(nn.Module):
#    def __init__(self):
#        super(SID, self).__init__()
#        
#    def forward(self, output, target, mask=None):
#        
#        output = torch.clamp(output, 0, 1)
#        
#        a1 = output * torch.log10((output + EPS) / (target + EPS))
#        a2 = target * torch.log10((target + EPS) / (output + EPS))
#        
#        if mask is not None:
#            a1 = a1 * mask
#            a2 = a2 * mask
#        
#        a1_sum = a1.sum(dim=3).sum(dim=2)
#        a2_sum = a2.sum(dim=3).sum(dim=2)
#        
#        errors = torch.abs(a1_sum + a2_sum)
#        
#        return torch.mean(errors)