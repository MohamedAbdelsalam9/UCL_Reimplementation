from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from Util.util import log


def copy_freeze(model):
    model_copy = deepcopy(model)
    for param in model_copy.parameters():
        param.requires_grad = False
    return model_copy


def task_train(train_data, valid_data, new_model, criterion, optimizer, config, wandb):
    best_model = deepcopy(new_model)
    old_model = copy_freeze(new_model)  # best model for the previous task
    config['best_valid_loss'] = 1e10

    for epoch in range(config['epoch'], config['epochs_per_task']):
        config['epoch'] = epoch
        log_dict = {}

        log_dict[f"ucl_loss_{config['task_id']}"] = \
            epoch_train(train_data, new_model, old_model, criterion, optimizer, config)
        log_dict[f"train_loss_{config['task_id']}"], log_dict[f"train_acc_{config['task_id']}"] = \
            eval(train_data, new_model, config)
        log_dict[f"valid_loss_{config['task_id']}"], log_dict[f"valid_acc_{config['task_id']}"] = \
            eval(valid_data, new_model, config)

        log(epoch, log_dict, wandb)

        if log_dict[f"valid_loss_{config['task_id']}"] < config['best_valid_loss']:
            config['best_valid_loss'] = log_dict[f"valid_loss_{config['task_id']}"]
            best_model = deepcopy(new_model)
            wandb.config.update({'best_valid_loss': config['best_valid_loss']}, allow_val_change=True)

    #reset the model parameters to the best performing model
    new_model.load_state_dict(best_model.state_dict())
    return new_model


def epoch_train(task_data, new_model, old_model, criterion, optimizer, config):
    new_model.train()
    ucl_loss = 0
    data_len = 0
    for minibatch_id, minibatch_x_, minibatch_y_ in iter(task_data):
        minibatch_x = minibatch_x_.to(config['device'])
        minibatch_y = minibatch_y_.to(config['device'])
        data_len += minibatch_x.shape[0]

        output = new_model(minibatch_x, sample=True)[config["task_id"]]

        if config['task_id'] > 0:
            loss = criterion(output, minibatch_y, old_model, new_model)
        else:
            loss = criterion(output, minibatch_y)
        ucl_loss += loss.item() * minibatch_x.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return ucl_loss / data_len


def eval(task_data, new_model, config):
    new_model.eval()
    loss = 0
    accuracy = 0
    data_len = 0
    with torch.no_grad():
        for minibatch_id, minibatch_x_, minibatch_y_ in iter(task_data):
            minibatch_x = minibatch_x_.to(config['device'])
            minibatch_y = minibatch_y_.to(config['device'])
            output = new_model(minibatch_x, sample=False)[config["task_id"]]
            data_len += minibatch_x.shape[0]

            loss += F.cross_entropy(output, minibatch_y, reduction="sum").item()
            _, predictions = torch.max(output, dim=-1)
            accuracy += torch.sum(predictions == minibatch_y).item()
    return loss / data_len, accuracy / data_len
