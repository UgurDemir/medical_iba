import torch.nn as nn
import torchvision

def _build_resnet(model, nclass, in_channel=3, out=None, b2m=False):
    seq = []

    # First layer configuration
    if in_channel != 3:
        removed_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channel, out_channels=removed_conv.out_channels, kernel_size=removed_conv.kernel_size, 
                                stride=removed_conv.stride, padding=removed_conv.padding, dilation=removed_conv.dilation, 
                                groups=removed_conv.groups, bias=removed_conv.bias, padding_mode=removed_conv.padding_mode)

    # Last layer configuration
    if nclass != model.fc.out_features:
        seq = [nn.Linear(model.fc.in_features, nclass)]
    else:
        seq = [model.fc]

    # Output activation
    if out is not None:
        seq += [nn.__dict__[out]()]

    if b2m:
        raise Exception('b2m is not supported yet')
        '''
        if out == 'Sigmoid':
            seq += [Binary2MutliClass()]
        else:
            seq += [nn.Sigmoid(), Binary2MutliClass()]
        '''


    model.fc = nn.Sequential(*seq)

    return model

def resnet18(nclass, istrained, in_channel, out, **kwargs):
    model = torchvision.models.resnet18(pretrained=istrained, **kwargs)
    model = _build_resnet(model, nclass, in_channel, out)
    return model

def resnet34(nclass, istrained, in_channel, out, **kwargs):
    model = torchvision.models.resnet34(pretrained=istrained, **kwargs)
    model = _build_resnet(model, nclass, in_channel, out)
    return model

def resnet50(nclass, istrained, in_channel, out, **kwargs):
    model = torchvision.models.resnet50(pretrained=istrained, **kwargs)
    model = _build_resnet(model, nclass, in_channel, out)
    return model

def resnet101(nclass, istrained, in_channel, out, **kwargs):
    model = torchvision.models.resnet101(pretrained=istrained, **kwargs)
    model = _build_resnet(model, nclass, in_channel, out)
    return model
