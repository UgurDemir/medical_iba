import torch

from IBA.pytorch import IBA, tensor_to_np_img
from IBA.utils import plot_saliency_map

from reader.loader import get_test_loader, get_train_loader
from nets import model
import configs
import matplotlib.pyplot as plt
import os
from os.path import join, exists
import argparse
import cv2
import numpy as np
import argparse

def show_cam_on_image(img, mask, path="./e/cam.jpg"):
    #print(path)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path, np.uint8(255 * cam))

def get_layer(net, arch, layer):
    if arch.startswith('dense'):
        if layer == 'denseblock1':
            feat_layer = net.model.features[4]
        elif layer == 'denseblock2':
            feat_layer = net.model.features[6]
        elif layer == 'denseblock3':
            feat_layer = net.model.features[8]
        elif layer == 'denseblock4':
            feat_layer = net.model.features[10]
        else:
            raise Exception('Unknown layer name {} for arch {}'.format(layer, arch))
    elif arch.startswith('nest') or arch.startswith('resnet'):
        if layer == 'layer1':
            feat_layer = net.model.layer1
        elif layer == 'layer2':
            feat_layer = net.model.layer2
        elif layer == 'layer3':
            feat_layer = net.model.layer3
        elif layer == 'layer4':
            feat_layer = net.model.layer4
        else:
            raise Exception('Unknown layer name {} for arch {}'.format(layer, arch))
    else:
        raise Exception('Unknown arch name')

    return feat_layer

def run(opt, layer, weight_postfix, n_samples, vis_out_dir, orig_out_dir):
    # Data Loader
    eval_loader = get_test_loader(**{**opt['dataset'], 'batch_size':1})
    train_loader = get_train_loader(**opt['dataset'])

    # Network configuration
    netconf = {k:v for k, v in opt['netA'].items() if k in ['dev', 'model']}
    if 'istrained' in netconf['model']: netconf['model']['istrained'] = False
    netA = model.build(name='netA', **netconf, resume=join(opt['modeldir'], 'chk_{}.pth.tar'.format(weight_postfix))).eval()
    print(netA)

    # Feature Layer Selection
    feat_layer = get_layer(netA, opt['netA']['model']['arch'], layer)
    iba = IBA(feat_layer)
    
    # IBA stats calculation
    iba.reset_estimate()
    iba.estimate(netA.model, train_loader, device=torch.device(netA.dev), n_samples=n_samples, progbar=True)

    for i, batch in enumerate(eval_loader):
        x, label = batch[0].to(netA.dev), batch[1].to(netA.dev)

        model_loss_closure = lambda x: -torch.log_softmax(netA(x), dim=1)[:, label].mean()
        heatmap = iba.analyze(x, model_loss_closure)

        # Visualization
        x = x.cpu()[0,:,:,:].permute(1,2,0).numpy()
        x = eval_loader.dataset.norm.scale_to_01(x)
        show_cam_on_image(x, heatmap, join(vis_out_dir, str(i)+ '_label_'+ str(label.item()) +'.png'))
        if orig_out_dir is not None:
            cv2.imwrite(join(orig_out_dir, str(i)+ '_label_'+ str(label.item()) +'.png'), np.uint8(255 * x))


if __name__ == "__main__":
    # Argumnent parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=str, required=True, help='Experiment name')
    parser.add_argument('-m', type=str, default='desktop', help='Machine name')
    parser.add_argument('-n', type=int, default=1000, help='# of samples for stat calculation')
    parser.add_argument('-w', type=str, default='best_v_accuracy', help='Postfix of the weight file')
    parser.add_argument('--layer', type=str, required=True, help='Feature layer name')
    parser.add_argument('--org', action='store_true', help='Save input images')
    args = parser.parse_args()
    
    # Configuration
    opt = configs.parse(args.m, args.e)

    # Create output directories
    vis_out_dir = join(opt['outdir'], 'vis', '[{}]_{}_iba'.format(args.w, args.layer))
    orig_out_dir = join(opt['outdir'], 'vis', 'org')
    if not exists(vis_out_dir): os.makedirs(vis_out_dir)
    if not exists(orig_out_dir) and args.org: os.makedirs(orig_out_dir)
    if not args.org: orig_out_dir = None
    run(opt, layer=args.layer, weight_postfix=args.w, n_samples=args.n, vis_out_dir=vis_out_dir, orig_out_dir=orig_out_dir)