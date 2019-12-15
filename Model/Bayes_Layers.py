import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal


class Gaussian():
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        # noise is always in between 0 and 1
        self.epsilon_noise = normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def sigma(self):
        sigma = torch.log1p(torch.exp(self.rho))
        return sigma

    def sample(self):
        return self.mu + self.sigma() * self.epsilon_noise.sample(torch.shape(self.mu))


class BayesLinear(nn.Module):
    def __init__(self, in_shape, out_shape, ratio):
        super(BayesLinear).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        self.weight_mu = nn.Parameter(torch.zeros((out_shape, in_shape)), requires_grad=True)
        self.weight_rho = nn.Parameter(torch.zeros((out_shape, 1)), requires_grad=True)

        #Gaussian object for the weight
        self.weight= Gaussian(self.weight_mu, self.weight_rho)


        
    def forward(self, input, sample=False):
        device = input.device

        if sample==False:
            weight= self.weight_mu
        else:
            weight= self.weight.sample()

#        output = torch.zeros((self.out_shape)).to(device)
        x= F.Linear(input, weight).to(device)
#        return output
        return x

    
#[128,245]
class BayesNet(nn.Module):
    def __init__(self, in_shape, out_shape, num_hidden_layers=1, hidden_sizes = [128], ratio=0.5):
        super(BayesNet).__init__()
        if len(hidden_sizes) == 1:
            self.hidden_sizes = [hidden_sizes[0] for i in range(num_hidden_layers)]
        else:
            assert (len(hidden_sizes) == num_hidden_layers), "You didn't specify the hidden shapes of all the layers"

        # default number of nodes for our task:  256. 2 layers

        self.hidden_sizes = hidden_sizes
        self.num_hidden_layers = num_hidden_layers

        self.layers= nn.ModuleList([ BayesLinear(hidden_sizes[i],hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])

        self.output_layer= BayesLinear(hidden_sizes[-1], 1)

        self.relu = torch.nn.ReLU()

    def forward(self, input, sample=False):
        x = self.layers[0](input)
        x= self.relu(x)
        for layer in self.layers[1:]:
            x = layer(x)
            x= self.relu(x)
        return x
