import importlib

from nets import densenet, resnest, resnet


def build_net(arch, **args):

    if arch.startswith('dense'):
        if arch.endswith('121'):
            n = densenet.dense121
        elif arch.endswith('169'):
            n = densenet.dense169
        elif arch.endswith('201'):
            n = densenet.dense201
    elif arch.startswith('nest'):
        if arch.endswith('50'):
            n = resnest.nest50
        elif arch.endswith('101'):
            n = resnest.nest101
        elif arch.endswith('200'):
            n = resnest.nest200
        elif arch.endswith('269'):
            n = resnest.nest269
    elif arch.startswith('resnet'):
        if arch.endswith('18'):
            n = resnet.resnet18
        elif arch.endswith('34'):
            n = resnet.resnet34
        elif arch.endswith('50'):
            n = resnet.resnet50
        elif arch.endswith('101'):
            n = resnet.resnet101
    else:
        n = importlib.import_module('nets.'+arch.lower()).__dict__[arch.upper()]

    return n(**args)
