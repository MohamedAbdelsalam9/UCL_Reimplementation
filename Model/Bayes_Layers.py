import torch
import torch.nn as nn
import numpy as np

class BayesLinear(nn.Module):
    def __init__(self, in_shape, out_shape, ratio):
        super(BayesLinear).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.weight_mu = nn.Parameter(torch.zeros((out_shape, in_shape)), requires_grad=True)
        self.weight_rho = nn.Parameter(torch.zeros((out_shape, 1)), requires_grad=True)

    def forward(self, input, sample=False):
        device = input.device
        output = torch.zeros((self.out_shape)).to(device)
        return output


class BayesNet(nn.Module):
    def __init__(self, in_shape, out_shape, num_hidden_layers=1, hidden_sizes = [128], ratio=0.5):
        super(BayesNet).__init__()
        if len(hidden_sizes) == 1:
            self.hidden_sizes = [hidden_sizes[0] for i in range(num_hidden_layers)]
        else:
            assert (len(hidden_sizes) == num_hidden_layers), "You didn't specify the hidden shapes of all the layers"
        self.hidden_sizes = hidden_sizes
        self.num_hidden_layers = num_hidden_layers
        self.layers = [BayesLinear(in_shape, out_shape) for i in num_hidden_layers]

    def forward(self, input, sample=False):
        x = self.layers[0](input)
        for layer in self.layers[1:]:
            x = layer(x)
        return x
