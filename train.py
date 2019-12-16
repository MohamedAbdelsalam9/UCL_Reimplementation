import sys
import torch
from torch.optim import Adam
import argparse
import os
import wandb
from Util.data_loader import BatchIterator, get_split_mnist
from Util.per_task_trainer import task_train, copy_freeze
from Util.util import print_msg
from Model.Bayes_Layers import BayesNet
from Model.Custom_Loss import UCLLoss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="split_mnist")  # split_mnist
parser.add_argument('--epochs_per_task', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--data_path', type=str, default="Dataset")
parser.add_argument('--wandb_project', type=str, default='ucl')
parser.add_argument('--beta', type=float, default=0.0001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_rho', type=float, default=0.001)
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=256)  # todo make it an array
# parser.add_argument('--optimizer', type=str, default='adam')

args = parser.parse_args()
sys.path.append(args.data_path)

config = {}
config['dataset'] = args.dataset
config['data_path'] = args.data_path
config['seed'] = args.seed
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['batch_size'] = args.batch_size
config['flatten'] = False
if args.dataset in ['split_mnist']:
    config['flatten'] = True
config['epochs_per_task'] = args.epochs_per_task
config['epoch'] = 0
config['task_id'] = 0
config["wandb_project"] = args.wandb_project

# config['optimizer'] = args.optimizer
config['num_hidden_layers'] = args.num_hidden_layers
config['hidden_size'] = [args.hidden_size]  # todo make it an array
config['beta'] = args.beta
config['lr'] = args.lr
config['lr_rho'] = args.lr_rho

if __name__ == '__main__':
    if args.dataset == 'split_mnist':
        data, taskcla, inputsize = get_split_mnist(datapath=config['data_path'], tasknum=5)

    if config['flatten']:
        input_size = inputsize[0] * inputsize[1] * inputsize[2]

    config['num_tasks'] = len(taskcla)
    config['classes_per_task'] = [c_t for _, c_t in taskcla]

    new_model = BayesNet(input_size, taskcla, num_hidden_layers=config['num_hidden_layers'],
                         hidden_sizes=config['hidden_size'], ratio=0.5)

    new_model.to(config['device'])
    # todo specify sigma_init
    criterion = UCLLoss(config['beta'], sigma_init=[0], num_layers=config['num_hidden_layers'])
    optimizer = Adam([
        {"params": [p for name, p in new_model.named_parameters() if "rho" not in name], "lr": config['lr']},
        {"params": [p for name, p in new_model.named_parameters() if "rho" in name], "lr": config['lr_rho']}],
        lr=config['lr'])

    print_msg(f"using {config['device']}\n")

    wandb.init(project=config["wandb_project"], allow_val_change=True)
    wandb.config.update(config, allow_val_change=True)
    wandb.watch(new_model)

    for task_id, num_classes in taskcla[config['task_id']:]:
        config['task_id'] = task_id
        config['epoch'] = 0
        config['best_valid_loss'] = 1e10
        wandb.config.update(config, allow_val_change=True)
        train_data = BatchIterator(data[task_id]['train'], config['batch_size'], shuffle=True,
                                   flatten=config['flatten'])
        valid_data = BatchIterator(data[task_id]['valid'], config['batch_size'], shuffle=False,
                                   flatten=config['flatten'])
        # get best model for the new task
        new_model = task_train(train_data, valid_data, new_model, criterion, optimizer, config, wandb)
        torch.save({'model_State_dict': new_model.state_dict(), 'config': config},
                   os.path.join(wandb.run.dir, f"best_model_task_{task_id}"))
