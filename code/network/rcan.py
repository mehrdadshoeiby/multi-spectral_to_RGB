#!/usr/local/bin/python
from network import common
import torch.nn as nn
import torch
import torchvision

def make_model(args, parent=False):
    return RCANResNet18(args)


########################## Main Network ##############################

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res
        
## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
#        n_resgroups = args.n_resgroups
#        n_resblocks = args.n_resblocks
#        n_feats = args.n_feats
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.n_reduction
        act = nn.ReLU()
        # change kernel_size (4) and stride (2) for different scale factor (2)
        self.decon3 = nn.ConvTranspose2d(in_channels=16, out_channels=64,
                                         kernel_size=3, stride=1, padding=1)
        # define head module
        modules_head = [conv(64, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act,
                res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module 
        # change 14 to 3 for RGB
        modules_tail = [
            conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.decon3(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x
        
#%%########################## Loss Network ##############################
# RCAN without CA
class RBN(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RBN, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class RGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(RGroup, self).__init__()
        modules_body = []
        modules_body = [
            RBN(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
class RIR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RIR, self).__init__()
        
#        n_resgroups = args.n_resgroups
#        n_resblocks = args.n_resblocks
#        n_feats = args.n_feats
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.n_reduction
        act = nn.ReLU()
        # change kernel_size (4) and stride (2) for different scale factor (2)
        self.decon3 = nn.ConvTranspose2d(in_channels=3, out_channels=64,
                                         kernel_size=3, stride=1, padding=1)
        # define head module
        modules_head = [conv(64, n_feats, kernel_size)]
        # define body module
        modules_body = [
            RGroup(
                conv, n_feats, kernel_size, reduction, act=act,
                res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        # define tail module 
        # change 14 to 3 for RGB
        modules_tail = [
            conv(n_feats, 256, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.decon3(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x
#%%############################## main + loss network  ##################
class RCANRIR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCANRIR, self).__init__()
        module_main = [RCAN(args)]
        self.main = nn.Sequential(*module_main)
        module_mask = [RIR(args)]       
        self.mask = nn.Sequential(*module_mask)
        self.deconv_last = nn.ConvTranspose2d(in_channels=259, out_channels=3,
                                         kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.main(x)
        z = self.mask(y)
        k = torch.cat((y,z),dim=1)
#        print("z size is: {}".format(z.shape))
        out = self.deconv_last(k)
        
        return out, y

#%% ############## loss network Resnet-18 ############################
#####################################################################
class ResNet18Mask(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ResNet18Mask, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.deconv_last = nn.ConvTranspose2d(in_channels=64, out_channels=256,
                                         kernel_size=6, stride=4, padding=1)
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.deconv_last(x)

        return x
#%% ############### main + loss network: RCAN + Resnet-18 ##############
class RCANResNet18(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCANResNet18, self).__init__()
        module_main = [RCAN(args)]
        self.main = nn.Sequential(*module_main)
        module_mask = [ResNet18Mask(args)]       
        self.mask = nn.Sequential(*module_mask)
        self.deconv_last = nn.ConvTranspose2d(in_channels=259, out_channels=3,
                                         kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.main(x)
        z = self.mask(y)
        k = torch.cat((y,z),dim=1)
#        print("z size is: {}".format(z.shape))
        out = self.deconv_last(k)
        
        return out, y
