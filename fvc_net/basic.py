import math
import time
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import imageio
import datetime

from fvc_net.models.utils import (conv,deconv,quantize_ste,update_registered_buffers,)
from PIL import Image


out_channel_F = 64
out_channel_O = 64
out_channel_E=96
out_channel_M = 128
deform_ks = 3
deform_groups = 8
out_channel_N=64
#
class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel,kernel_size, stride, padding=kernel_size // 2)
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel,kernel_size, stride, padding=kernel_size // 2)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)

    def forward(self, x):
        firstlayer = self.conv1(x)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        return x + seclayer

def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )




class feature_reconsnet(nn.Module):
    def __init__(self):
        super(feature_reconsnet, self).__init__()
        self.resb1 = ResBlock(out_channel_F, out_channel_F)
        self.resb2 = ResBlock(out_channel_F, out_channel_F)
        self.resb3 = ResBlock(out_channel_F, out_channel_F)
        self.deconv1= deconv(out_channel_F, 3, kernel_size=5, stride=2)
    def forward(self,x):
        x1=self.resb1(x)
        x2=self.resb2(x1)
        x3=self.resb3(x2)
        x4=self.deconv1(x3+x)
        return x4


def geti(lamb):
    if lamb == 2048:
        return 'H265L20'
    elif lamb == 1024:
        return 'H265L23'
    elif lamb == 512:
        return 'H265L26'
    elif lamb == 256:
        return 'H265L29'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)

def get_chen2020anchor_i(lamb):
    if lamb == 2048:
        return 'chen2020anchorL6'
    elif lamb == 1024:
        return 'chen2020anchorL5'
    elif lamb == 512:
        return 'chen2020anchorL4'
    elif lamb == 256:
        return 'chen2020anchorL3'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)


def get_bmshj2018_hyperprior_i(lamb):
    # if lamb == 2048:
    #     return 'bmshj2018_hyperpriorL8'
    if lamb == 2048:
        return 'bmshj2018_hyperpriorL8'
    elif lamb == 1024:
        return 'bmshj2018_hyperpriorL6'
    elif lamb == 512:
        return 'bmshj2018_hyperpriorL4'
    elif lamb == 256:
        return 'bmshj2018_hyperpriorL2'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)


def get_mbt2018_i(lamb):
    if lamb == 32:
        return 'mbt2018L8'
    # elif lamb == 32:
    #     return 'mbt2018L6'
    elif lamb == 16:
        return 'mbt2018L4'
    elif lamb == 18:
        return 'mbt2018L2'
    elif lamb == 2048:
        return 'mbt2018L8'
    elif lamb == 1024:
        return 'mbt2018L6'
    elif lamb == 512:
        return 'mbt2018L4'
    elif lamb == 256:
        return 'mbt2018L2'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)


def read_frame_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)/255
    return input_image

def write_torch_frame(frame, path):
    frame_result = frame.clone()
    frame_result = frame_result.cpu().detach().numpy().transpose(1, 2, 0)*255
    frame_result = np.clip(np.rint(frame_result), 0, 255)
    frame_result = Image.fromarray(frame_result.astype('uint8'), 'RGB')
    frame_result.save(path)

# def imagepadding(x):
#     h, w = x.size(2), x.size(3)
#     p = 64  # maximum 6 strides of 2
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top
#     x_padded = F.pad(
#         x,
#         (padding_left, padding_right, padding_top, padding_bottom),
#         mode="constant",
#         value=0,
#     )
#     return x_padded,padding_left,padding_right,padding_top,padding_bottom


def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature
