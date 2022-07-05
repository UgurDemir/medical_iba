import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision


def dense121(nclass, istrained, in_channel=3, **kwargs):
    model = torchvision.models.densenet121(pretrained=istrained, **kwargs)

    # First layer configuration
    if in_channel != 3:
        conv1 = list(model.features.children())#[1:]
        removed_conv = conv1[0]
        new_in_conv = nn.Conv2d(in_channel, out_channels=removed_conv.out_channels, kernel_size=removed_conv.kernel_size, 
                                stride=removed_conv.stride, padding=removed_conv.padding, dilation=removed_conv.dilation, 
                                groups=removed_conv.groups, bias=removed_conv.bias, padding_mode=removed_conv.padding_mode)
        model.features = nn.Sequential(new_in_conv, *conv1[1:])


    if nclass != model.classifier.out_features:
        model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, nclass), nn.Sigmoid())
    return model

def dense169(nclass, istrained, **kwargs):
    model = torchvision.models.densenet161(pretrained=istrained, **kwargs)
    if nclass != model.classifier.out_features:
        model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, nclass), nn.Sigmoid())
    return model

def dense201(nclass, istrained, **kwargs):
    model = torchvision.models.densenet201(pretrained=istrained, **kwargs)
    if nclass != model.classifier.out_features:
        model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, nclass), nn.Sigmoid())
    return model

"""
class DenseNet121(nn.Module):
    def __init__(self, nclass, istrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=istrained)
        kernelCount = self.densenet121.classifier.in_features
        # self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        self.classify = nn.Sequential(nn.Linear(kernelCount, nclass), nn.Sigmoid())

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classify(out)
        return out


class DenseNet169(nn.Module):
    def __init__(self, nclass, istrained):
        super(DenseNet169, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=istrained)
        kernelCount = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, nclass), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    
class DenseNet201(nn.Module):
    def __init__ (self, nclass, istrained):
        super(DenseNet201, self).__init__()
        self.densenet201 = torchvision.models.densenet201(pretrained=istrained)
        kernelCount = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, nclass), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x

"""