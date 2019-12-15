import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


#Todo placeholder
class UCLLoss(nn.Module):
    def __init__(self, beta=0.0001):
        super(UCLLoss, self).__init__()
        self.beta = beta

    def forward(self, output, target, new_model=None, old_model=None):
        if old_model is not None and new_model is not None:
            return self.nll_loss(output, target) + self.regularizer(new_model, old_model)

    def nll_loss(self, output, target):
        return F.cross_entropy(output, target, reduction="mean")

    def regularizer(selfSelf, new_model, old_model):
            return 0 #todo