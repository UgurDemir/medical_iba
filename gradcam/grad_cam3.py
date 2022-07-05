import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import models

class DenseNetCAMExtractor():
    def __init__(self, model, target_block, target_layer):
        self.model = model
        self.target_block = target_block
        self.target_layer = target_layer
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def _extract_feats(self, feature_module, x):
        outputs = []
        self.gradients = []

        features = [x]
        for name, layer in feature_module._modules.items():
            new_features = layer(features)
            features.append(new_features)
            if name == self.target_layer:
                new_features.register_hook(self.save_gradient)
                outputs += [new_features]
        return outputs, torch.cat(features, 1)

    def _extract_feats22(self, feature_module, x):
        outputs = []
        self.gradients = []
        for name, module in feature_module._modules.items():
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

    def __call__(self, x):
        target_activations = []
        for name, module in self.model.features._modules.items():
            if name == self.target_block:
                target_activations, x = self._extract_feats(module, x)
            else:
                x = module(x)
        
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)

        return target_activations, out




class ResNetCAMExtractor():
    def __init__(self, model, target_block, target_layer):
        self.model = model
        self.target_block = target_block
        self.target_layer = target_layer
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def _extract_feats(self, feature_module, x):
        outputs = []
        self.gradients = []
        for name, module in feature_module._modules.items():
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if name == self.target_block:
                target_activations, x = self._extract_feats(module, x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x



class GradCam:
    def __init__(self, cam_extractor):
        self.extractor = cam_extractor
        self.extractor.model.eval()        

    def forward(self, input):
        return self.extractor.model(input)

    def __call__(self, input, index=None):
        features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(input.device) * output)

        #self.extractor.feature_module.zero_grad()
        self.extractor.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
