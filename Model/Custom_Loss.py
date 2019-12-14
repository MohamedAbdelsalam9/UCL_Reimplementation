import torch
import torch.nn as nn
import numpy as np


#Todo placeholder
class UCLLoss(nn.Module):
    def __init__(self, beta=0.0001):
        super(UCLLoss, self).__init__()
        self.beta = beta

    def forward(self, old_model, new_model):
        pass