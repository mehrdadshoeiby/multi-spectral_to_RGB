# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:53:54 2019

@author: sho092
"""

import torch 
import torchvision

model = torchvision.models.resnet18(pretrained=True)
print(model)
x = torch.randn(1, 3,1000, 2000)
x = model.conv1(x)
x = model.bn1(x)
x = model.relu(x)
x = model.maxpool(x)
x = model.layer1(x)
#x = model.layer2(x)
#x = model.layer3(x)
#x = model.layer4(x)
print(x.shape)