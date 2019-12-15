import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


#Todo placeholder
class UCLLoss(nn.Module):
    def __init__(self, beta=0.0001, sigma_init=[0], num_layers=1):
        super(UCLLoss, self).__init__()
        self.beta = beta
        if len(sigma_init) > 1:
            assert (len(sigma_init) == num_layers), "you didn't specify a sigma_init for all the layers"
            self.sigma_init = sigma_init
        else:
            self.sigma_init = [sigma_init[0] for _ in range(num_layers)]

    def forward(self, output, target, new_model=None, old_model=None):
        if old_model is not None and new_model is not None:
            return self.nll_loss(output, target) + self.regularizer(new_model, old_model)

    def nll_loss(self, output, target):
        return F.cross_entropy(output, target, reduction="mean")

    def regularizer(selfSelf, new_model, old_model):
            return 0 #todo