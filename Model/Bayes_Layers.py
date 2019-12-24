import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
import math


class GaussianSampler(object):
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.epsilon_noise = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def sigma(self):
        sigma = torch.log1p(torch.exp(self.rho))
        return sigma

    def sample(self):
        return self.mu + self.sigma() * self.epsilon_noise.sample(self.mu.shape).squeeze()


class BayesLinear(nn.Module):
    def __init__(self, in_shape, out_shape, ratio):
        super(BayesLinear, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.bias = nn.Parameter(torch.zeros(out_shape))
        self.ratio = ratio

        var = 2 / in_shape
        ratio_var = var * ratio
        mu_var = var - ratio_var
        noise_std, mu_std = math.sqrt(ratio_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std) - 1)

        self.weight_mu = nn.Parameter(torch.zeros((out_shape, in_shape)), requires_grad=True)
        nn.init.uniform_(self.weight_mu, -bound, bound)

        self.weight_rho = nn.Parameter(torch.zeros((out_shape, 1)), requires_grad=True)
        nn.init.uniform_(self.weight_rho, rho_init, rho_init)

        # Gaussian sampler for the weight and bias
        self.weight_sampler = GaussianSampler(self.weight_mu, self.weight_rho)

    def forward(self, input_data, sample=False):
        if not sample:
            weight = self.weight_mu
            bias = self.bias
        else:
            weight = self.weight_sampler.sample()
            bias = self.bias
        x = F.linear(input_data, weight, bias)
        return x


class BayesNet(nn.Module):
    def __init__(self, input_shape, task_cla, num_hidden_layers=1, hidden_sizes=(128,), ratio=0.5):
        super(BayesNet, self).__init__()
        if len(hidden_sizes) == 1:
            self.hidden_sizes = [hidden_sizes[0] for _ in range(num_hidden_layers)]
        else:
            assert (len(hidden_sizes) == num_hidden_layers), "You didn't specify the hidden shapes of all the layers"
            self.hidden_sizes = hidden_sizes

        self.num_hidden_layers = num_hidden_layers
        self.task_cla = task_cla
        self.ratio = ratio

        self.layers = nn.ModuleList([BayesLinear(input_shape, hidden_sizes[0], ratio)])
        self.layers.extend(nn.ModuleList([BayesLinear(self.hidden_sizes[i], self.hidden_sizes[i + 1], ratio)
                                          for i in range(self.num_hidden_layers - 1)]))

        # the multi-head output layer
        self.output_layer = nn.ModuleList(
            [nn.Linear(hidden_sizes[-1], num_class) for _, num_class in self.task_cla])

        self.relu = torch.nn.ReLU()

    def forward(self, input, sample=False):
        x = self.layers[0](input)
        x = self.relu(x)
        for layer in self.layers[1:]:
            x = layer(x)
            x = self.relu(x)

        # the final output per task (multi_head)
        output = [self.output_layer[t](x) for t in range(len(self.task_cla))]
        return output
