import torch
import torch.nn as nn


# b2m: True|False, convert binary output to multi class
def _build_resnest(model, nclass, in_channel=3, out=None, b2m=False):
    seq = []

    # First layer configuration
    if in_channel != 3:
        conv1 = list(model.conv1.children())  # [1:]
        removed_conv = conv1[0]
        new_in_conv = nn.Conv2d(in_channel, out_channels=removed_conv.out_channels, kernel_size=removed_conv.kernel_size, 
                                stride=removed_conv.stride, padding=removed_conv.padding, dilation=removed_conv.dilation, 
                                groups=removed_conv.groups, bias=removed_conv.bias, padding_mode=removed_conv.padding_mode)
        model.conv1 = nn.Sequential(new_in_conv, *conv1[1:])

    # Last layer configuration
    if nclass != model.fc.out_features:
        seq = [nn.Linear(model.fc.in_features, nclass)]
    else:
        seq = [model.fc]

    # Output activation
    if out is not None:
        seq += [nn.__dict__[out]()]

    if b2m:
        if out == 'Sigmoid':
            seq += [Binary2MutliClass()]
        else:
            seq += [nn.Sigmoid(), Binary2MutliClass()]


    model.fc = nn.Sequential(*seq)

    return model

class Binary2MutliClass(nn.Module):
    # def __init__(self):
    #    super(DenseNet121, self).__init__()

    # Expect binary input: x [b, 1] --> [b, 2]
    def forward(self, x):
        o = torch.cat([1.0-x, x], dim=1)
        return o

def nest50(nclass, istrained, in_channel, out, b2m=False, **kwargs):
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=istrained, **kwargs)
    model = _build_resnest(model, nclass, in_channel, out, b2m)
    return model

def nest101(nclass, istrained, in_channel, out, b2m=False, **kwargs):
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=istrained, **kwargs)
    model = _build_resnest(model, nclass, in_channel, out, b2m)
    return model


def nest200(nclass, istrained, in_channel, out, b2m=False, **kwargs):
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=istrained, **kwargs)
    model = _build_resnest(model, nclass, in_channel, out, b2m)
    return model

def nest269(nclass, istrained, in_channel, out, b2m=False, **kwargs):
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=istrained, **kwargs)
    model = _build_resnest(model, nclass, in_channel, out, b2m)
    return model
