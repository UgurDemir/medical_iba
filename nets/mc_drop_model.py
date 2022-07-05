import torch
import torch.nn as nn


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class MCDropoutModel(nn.Module):
    # n: number of sample that is calculated for MCDroput
    def __init__(self, model, n=10):
        super(MCDropoutModel, self).__init__()
        self.model = model
        self.n = n

        enable_dropout(self.model)
    
    def forward(self, x):
        preds = []
        for _ in range(self.n):
            y = self.model(x)
            preds.append(y)

        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std
