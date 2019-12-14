import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
import wandb
from Util.split_mnist_loader import get_data, BatchIterator
from Model.Bayes_Layers import BayesNet
from Model.Custom_Loss import UCLLoss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="split_mnist")
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--data_path', type=str, default="Dataset")
parser.add_argument('--wandb_project', type=str, default='ucl')

args = parser.parse_args()
sys.path.append(args.data_path)

config = {}
config['dataset'] = args.dataset
config['data_path'] = args.data_path
config['seed'] = args.seed
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['batch_size'] = args.batch_size
config['flatten'] = False
if args.dataset == 'split_mnist':
    config['flatten'] = True


if __name__ == '__main__':
    data, taskcla, inputsize = get_data(seed=config['seed'], datapath=config['data_path'], tasknum=5)
    if config['flatten']:
        input_size = inputsize[0] * inputsize[1] * inputsize[2]

    # model = BayesNet(input_size, taskcla, num_hidden_layers=1, hidden_sizes=[128], ratio=0.5)
    model = BayesNet(input_size, 10, num_hidden_layers=1, hidden_sizes=[128], ratio=0.5)
    model.to(config['device'])

    config['num_tasks'] = len(taskcla)
    config['epochs_per_task'] = int(args.num_epochs / len(taskcla))
    config['num_epochs'] = config['epochs_per_task'] * config['num_tasks']
    config['epoch'] = 0
    config['task_id'] = 0

    epochs_per_task = int(args.num_epochs / len(taskcla))
    for task_id, num_classes in taskcla[config['task_id']:]:
        config['task_id'] = task_id
        task_data = BatchIterator(data[task_id]['train'], config['batch_size'], flatten=config['flatten'])
        for epoch in range(config['epoch'], config['epochs_per_task']):
            config['epoch'] = epoch
            for minibatch_id, minibatch_x_, minibatch_y_ in iter(task_data):
                minibatch_x = minibatch_x_.to(config['device'])
                minibatch_y = minibatch_y_.to(config['device'])
            pass
        pass
