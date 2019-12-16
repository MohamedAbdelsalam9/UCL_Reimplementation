import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from Model import Bayes_Layers
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

    def regularizer(self, new_model, old_model):

        L2_mu_regularizer_sum = 0
        L1_regularizer_magnitude_learnt_weights=0
        regularizer_third_term=0

        L2_bias_reg_sum=0
        L1_bias_regularizer_sum=0

        regularization_total=0

        #output layer is not bayesian. So this regularizer does not apply to it
        for i, old_model_layer,  current_model_layer in enumerate(zip(old_model.layers(), new_model.layers())):
            if not isinstance(current_model_layer, Bayes_Layers.BayesLinear):
                continue

            current_weight_mu = current_model_layer.weight_mu
            old_weight_mu = old_model_layer.weight_mu

            current_bias = current_model_layer.bias
            old_bias = old_model_layer.bias

            #careful sigma init. not var
            sigma_init_layer= self.sigma_init[i]
            if (i-1)>=0:
                sigma_init_layer_previous= self.sigma_init[i-1]
            else:
                sigma_init_layer_previous=torch.zeros(sigma_init_layer.shape)

            #sigma old task current layer l
            sigma_old_task_current_layer= torch.log1p(torch.exp(old_model_layer.rho))
            #to get sigma old task, for layer l-1
            if (i-1)>=0:
                sigma_old_task_previous_layer= torch.log1p(torch.exp(old_model.layers()[i-1].rho))
                strength_outgoing_from_current_node = sigma_init_layer / sigma_old_task_current_layer
                strength_incoming = sigma_init_layer_previous / sigma_old_task_previous_layer
                regularization_strength_for_weight = torch.max(strength_outgoing_from_current_node, strength_incoming)
            else:
                regularization_strength_for_weight= sigma_init_layer / sigma_old_task_current_layer

            mu_diff= current_weight_mu - old_weight_mu

            #Is some broadcasting required here, to do hadamard product??
            #also make sure that this is L2 norm
            term1= regularization_strength_for_weight * (mu_diff)
            L2_mu_regularizer_sum+= 0.5 * term1.norm(2)

            #make sure term2 is L1 norm
            term2= (( old_weight_mu/ sigma_old_task_current_layer)**2) * (mu_diff)
            L1_regularizer_magnitude_learnt_weights+= ((sigma_init_layer)**2)* term2.norm(1)

            simga_current_task_layer= torch.log1p(torch.exp(current_model_layer.rho))
            sigma_ratio_current_to_old= simga_current_task_layer/sigma_old_task_current_layer
            term3= (sigma_ratio_current_to_old**2) - torch.log((sigma_ratio_current_to_old**2))

            term4= (simga_current_task_layer**2) - torch.log((simga_current_task_layer**2))

            ones= torch.t(torch.ones(simga_current_task_layer.shape))

            regularizer_third_term+= self.beta * ones * (term3 + term4)

            #for bias here
            bias_regularization_each_node= torch.squeeze(strength_outgoing_from_current_node)
            bias_sigma= torch.squeeze(sigma_old_task_current_layer)
            L2_mu_bias_reg = (bias_regularization_each_node * (current_bias - old_bias)).norm(2)
            L2_bias_reg_sum+= L2_mu_bias_reg

            L1_reg_bias= (sigma_init_layer**2)*((old_bias/bias_sigma)**2 * (current_bias - old_bias )).norm(1)
            L1_bias_regularizer_sum+= L1_reg_bias

        regularization_total= L2_mu_regularizer_sum + L1_regularizer_magnitude_learnt_weights + regularizer_third_term + L2_bias_reg_sum + L1_bias_regularizer_sum

        return regularization_total