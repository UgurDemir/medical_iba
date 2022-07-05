from traceback import print_tb
import torch

from os.path import join, isfile, basename, exists
import argparse
from tqdm import tqdm

from gradcam import grad_cam3
import info_bottleneck
from attribution_bottleneck.bottleneck.estimator import ReluEstimator
from attribution_bottleneck.attribution.per_sample_bottleneck import PerSampleBottleneckReader

from config import MODEL_CONFIGS
from nets.model import build_net
from preprocess import get_xform
from dataloader import FolderIterator


def load_model(model_arch, model_path, dev):
    netA = build_net(**model_arch).to(dev)
    checkpoint = torch.load(model_path, map_location=dev)
    netA.load_state_dict(checkpoint['netA']['model'])
    netA = netA.eval()

    return netA

def predict_with_gcam(netA, data_iter, dev, layer, sub_layer, target_classes):
    # Feature Layer Selection
    if type(netA).__name__ == 'ResNet':
        cam_extractor = grad_cam3.ResNetCAMExtractor(model=netA, target_block=layer, target_layer=sub_layer)
    elif type(netA).__name__ == 'DenseNet':
        cam_extractor = grad_cam3.DenseNetCAMExtractor(model=netA, target_block=layer, target_layer=sub_layer)
    else:
        raise Exception('Unknown network architecute {}'.format(type(netA)))
    grad_cam = grad_cam3.GradCam(cam_extractor)

    # Calculate gradcam
    for x, output_handler in data_iter:
        x = x.to(dev)

        #y = netA(x)
        for target in target_classes:
            heatmap2d = grad_cam(x, target)
            output_handler(target, heatmap2d)


def predict_with_iba(netA, arch_name, data_iter, dev, layer, target_classes, estim_file):
    # IBA initialization
    estim_state_dict = torch.load(estim_file)
    feat_layer = info_bottleneck.get_layer(netA, arch_name, layer)
    estim = ReluEstimator(feat_layer)
    estim.N = estim_state_dict['N']
    estim.S = estim_state_dict['S']
    estim.M = estim_state_dict['M']
    estim.num_seen = estim_state_dict['num_seen']
    reader = PerSampleBottleneckReader(netA, estim, progbar=False)

    for x, output_handler in data_iter:
        x = x.to(dev)

        #y = netA(x)
        for target in target_classes:
            heatmap2d = reader.heatmap(x, torch.LongTensor([target]).to(dev))
            output_handler(target, heatmap2d)

def main(attr, input_dir, output_dir, arch, device, scale_size, norm, target_classes, layer, sub_layer=None):
    # Params
    model_arch = MODEL_CONFIGS[arch]['model']
    model_path = MODEL_CONFIGS[arch]['weight']
    estim_file = MODEL_CONFIGS[arch]['estim']
    dev = torch.device(device)
    preproc = {'scale_size':scale_size, 'crop_size':scale_size, 'norm':norm}

    # Build network
    netA = load_model(model_arch, model_path, dev)

    # Create data iterator
    xform = get_xform(**preproc)
    folder_iter = FolderIterator(input_dir, output_dir, xform)

    if attr == 'iba':
        predict_with_iba(netA, model_arch['arch'], folder_iter, dev, layer, target_classes, estim_file)
    elif attr == 'gcam':
        predict_with_gcam(netA, folder_iter, dev, layer, sub_layer, target_classes)
    else:
        raise NotImplementedError('Unknown attribution method: {}'.format(attr))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Attribution method selection
    parser.add_argument('-a', choices=['iba', 'gcam'], required=True, help='Select the attribution type')
    
    # Predifined network selection
    parser.add_argument('--conf', type=str, required=True, help='Network architecture')

    # Feature layer: resnet, nest -> [..., layer3, ...] | dense -> [..., denseblock4, ...]
    parser.add_argument('--layer', type=str, required=True, help='Feature layer name')

    # [GradCAM option] Feature sub_layers: resnet, nest -> [1, 2, 3, ...] | dense -> [..., denselayer15 ...]
    parser.add_argument('--sub_layer', type=str, default=None, help='Sub-Layer name in the selected layer for gradcam')

    # Device selection
    parser.add_argument('--device', type=str, default='cuda:0', help='Device selection (cuda:0|cpu)')

    # Scale size
    parser.add_argument('--scale_size', type=int, default=512, help='Images are scaled to this size')

    # Intensity Normalization
    parser.add_argument('--norm', type=str, default='01', help='Intensity normalization')

    # Target Classes
    parser.add_argument('-t', nargs='+', required=True, help='Target Classes')

    # Input file or directory path
    parser.add_argument('--inp', type=str, default='_input', help='Input file or directory path')

    # Output directory
    parser.add_argument('--out', type=str, default='_output', help='Output directory')

    # Get the options
    args = parser.parse_args()
    print(args)


    target_classes = [int(tstr) for tstr in args.t]

    main(attr=args.a, input_dir=args.inp, output_dir=args.out, arch=args.conf, device=args.device, scale_size=args.scale_size, norm=args.norm, target_classes=target_classes, layer=args.layer, sub_layer=args.sub_layer)