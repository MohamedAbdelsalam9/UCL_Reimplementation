import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import Bayes_Layers


class UCLLoss(nn.Module):
    def __init__(self, beta=0.0001, sigma_init=(0.06,), num_layers=1):
        super(UCLLoss, self).__init__()
        self.beta = beta
        if len(sigma_init) > 1:
            assert (len(sigma_init) == num_layers), "you didn't specify a sigma_init for all the layers"
            self.register_buffer('sigma_init', torch.tensor(sigma_init))
        else:
            self.register_buffer('sigma_init', torch.tensor([sigma_init[0] for _ in range(num_layers)]))

    def forward(self, output, target, new_model=None, old_model=None):
        if old_model is not None and new_model is not None:
            return (self.nll_loss(output, target) / output.shape[0]) + self.regularizer(new_model, old_model)
        else:
            return self.nll_loss(output, target) / output.shape[0]

    def nll_loss(self, output, target):
        return F.cross_entropy(output, target, reduction="sum")

    def regularizer(self, new_model, old_model):

        l2_mu_regularizer_sum = 0
        l2_bias_reg_sum = 0
        l1_regularizer_magnitude_learnt_weights = 0
        l1_bias_regularizer_sum = 0
        regularizer_third_term = 0

        # output layer is not bayesian. So this regularizer does not apply to it
        for i, (old_model_layer, current_model_layer) in enumerate(zip(old_model.layers, new_model.layers)):
            if isinstance(current_model_layer, Bayes_Layers.BayesLinear):
                current_weight_mu = current_model_layer.weight_mu
                old_weight_mu = old_model_layer.weight_mu

                current_bias = current_model_layer.bias
                old_bias = old_model_layer.bias

                # careful sigma init. not var
                sigma_init_cur_layer = self.sigma_init[i]
                sigma_old_task_cur_layer = torch.log1p(torch.exp(old_model_layer.weight_rho))
                strength_old_task_cur_layer = sigma_init_cur_layer / sigma_old_task_cur_layer

                # to get sigma old task, for layer l-1
                if (i - 1) >= 0:
                    sigma_init_layer_previous = self.sigma_init[i - 1]
                    sigma_old_task_prev_layer = torch.log1p(torch.exp(old_model.layers[i - 1].weight_rho))
                    strength_old_task_prev_layer = sigma_init_layer_previous / sigma_old_task_prev_layer
                    regularization_strength_weight = torch.max(strength_old_task_cur_layer,
                                                               strength_old_task_prev_layer)
                else:
                    regularization_strength_weight = strength_old_task_cur_layer

                mu_diff = current_weight_mu - old_weight_mu

                term1 = regularization_strength_weight * (mu_diff)
                l2_mu_regularizer_sum += term1.norm(2) ** 2

                term2 = ((old_weight_mu / sigma_old_task_cur_layer) ** 2) * (mu_diff)
                l1_regularizer_magnitude_learnt_weights += (sigma_init_cur_layer ** 2) * term2.norm(1)

                sigma_current_task_layer = torch.log1p(torch.exp(current_model_layer.weight_rho))
                sigma_ratio_current_to_old = sigma_current_task_layer / sigma_old_task_cur_layer
                term3 = (sigma_ratio_current_to_old ** 2) - 2 * torch.log(sigma_ratio_current_to_old)

                term4 = (sigma_current_task_layer ** 2) - 2 * torch.log(sigma_current_task_layer)

                regularizer_third_term += (term3 + term4).sum()

                # for bias here
                regularization_strength_bias = torch.squeeze(strength_old_task_cur_layer)
                bias_sigma_old_task_cur_layer = torch.squeeze(sigma_old_task_cur_layer)
                L2_mu_bias_reg = (regularization_strength_bias * (current_bias - old_bias)).norm(2) ** 2
                l2_bias_reg_sum += L2_mu_bias_reg

                L1_reg_bias = (sigma_init_cur_layer ** 2) * (
                        (old_bias / bias_sigma_old_task_cur_layer) ** 2 * (current_bias - old_bias)).norm(1)
                l1_bias_regularizer_sum += L1_reg_bias

        regularization_total = 0.5 * (l2_mu_regularizer_sum + l2_bias_reg_sum) + (
                l1_regularizer_magnitude_learnt_weights + l1_bias_regularizer_sum) + (
                0.5 * self.beta * regularizer_third_term)

        return regularization_total
