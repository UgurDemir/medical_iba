
def get_layer(net, arch, layer):
    if arch.startswith('dense'):
        if layer == 'denseblock1':
            feat_layer = net.features[4]
        elif layer == 'denseblock2':
            feat_layer = net.features[6]
        elif layer == 'denseblock3':
            feat_layer = net.features[8]
        elif layer == 'denseblock4':
            feat_layer = net.features[10]
        else:
            raise Exception('Unknown layer name {} for arch {}'.format(layer, arch))
    elif arch.startswith('nest') or arch.startswith('resnet'):
        if layer == 'layer1':
            feat_layer = net.layer1
        elif layer == 'layer2':
            feat_layer = net.layer2
        elif layer == 'layer3':
            feat_layer = net.layer3
        elif layer == 'layer4':
            feat_layer = net.layer4
        else:
            raise Exception('Unknown layer name {} for arch {}'.format(layer, arch))
    else:
        raise Exception('Unknown arch name')

    return feat_layer

