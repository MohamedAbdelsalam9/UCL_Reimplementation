import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
import wandb
from Util.split_mnist_loader import get_data

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--data_path', type=str, default="Dataset")
parser.add_argument('--wandb_project', type=str, default='ucl')

args = parser.parse_args()
sys.path.append(args.data_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {}
config['data_path'] = args.data_path
config['seed'] = args.seed
config['device'] = device

if __name__ == '__main__':
    data, taskcla, inputsize = get_data(seed=args.seed, datapath=args.data_path, tasknum=5)
    pass