from copy import deepcopy
import torch
import torch.nn.functional as F
from Util.util import log, log_task, copy_freeze
from Util.data_loader import BatchIterator
from torch.optim import Adam


def task_train(train_data, valid_data, new_model, criterion, config, wandb):
    best_model = deepcopy(new_model)
    old_model = copy_freeze(new_model)  # best model for the previous task
    config['best_valid_loss'] = 1e10

    optimizer = Adam([
        {"params": [p for name, p in new_model.named_parameters() if "rho" not in name], "lr": config['lr']},
        {"params": [p for name, p in new_model.named_parameters() if "rho" in name], "lr": config['lr_rho']}],
        lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [i for i in range(5, 26, 5)], gamma=1/config['lr_factor'])

    for epoch in range(config['epoch'], config['epochs_per_task']):
        config['epoch'] = epoch
        log_dict = {}

        log_dict[f"ucl_loss_{config['task_id']}"] = \
            epoch_train(train_data, criterion, optimizer, config, new_model=new_model, old_model=old_model)
        log_dict[f"train_loss_{config['task_id']}"], log_dict[f"train_acc_{config['task_id']}"] = \
            eval(train_data, config['task_id'], new_model, config['device'])
        log_dict[f"valid_loss_{config['task_id']}"], log_dict[f"valid_acc_{config['task_id']}"] = \
            eval(valid_data, config['task_id'], new_model, config['device'])

        scheduler.step()
        log(epoch, log_dict, wandb)

        if log_dict[f"valid_loss_{config['task_id']}"] < config['best_valid_loss']:
            config['best_valid_loss'] = log_dict[f"valid_loss_{config['task_id']}"]
            best_model = deepcopy(new_model)
            wandb.config.update({'best_valid_loss': config['best_valid_loss']}, allow_val_change=True)

    # reset the model parameters to the best performing model
    new_model.load_state_dict(best_model.state_dict())
    return new_model


def task_eval(tasks_data, model, config, wandb):
    # log the accuracies of the new model on all observed tasks
    acc_dict = {}
    for task_id in range(config['task_id'] + 1):
        test_data = BatchIterator(tasks_data[task_id]['test'], config['batch_size'], shuffle=False,
                                  flatten=config['flatten'])
        _, acc_dict[f"task_{task_id}_test_acc"] = \
            eval(test_data, task_id, model, config['device'])
    acc_dict[f"average_test_acc"] = sum(acc_dict.values()) / len(acc_dict)
    log_task(config['task_id'], acc_dict, wandb)


def epoch_train(task_data, criterion, optimizer, config, new_model, old_model):
    new_model.train()
    ucl_loss = 0
    data_len = 0
    for minibatch_id, minibatch_x_, minibatch_y_ in iter(task_data):
        minibatch_x = minibatch_x_.to(config['device'])
        minibatch_y = minibatch_y_.to(config['device'])
        data_len += minibatch_x.shape[0]

        output = new_model(minibatch_x, sample=True)[config["task_id"]]

        if config['no_ucl_reg'] or config['task_id'] == 0:
            loss = criterion(output, minibatch_y)  # no regularizer
        else:
            loss = criterion(output, minibatch_y, new_model=new_model, old_model=old_model)

        ucl_loss += loss.item() * minibatch_x.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return ucl_loss / data_len


def eval(task_data, task_id, new_model, device):
    new_model.eval()
    loss = 0
    accuracy = 0
    data_len = 0
    with torch.no_grad():
        for minibatch_id, minibatch_x_, minibatch_y_ in iter(task_data):
            minibatch_x = minibatch_x_.to(device)
            minibatch_y = minibatch_y_.to(device)
            output = new_model(minibatch_x, sample=False)[task_id]
            data_len += minibatch_x.shape[0]

            loss += F.cross_entropy(output, minibatch_y, reduction="sum").item()
            _, predictions = torch.max(output, dim=-1)
            accuracy += torch.sum(predictions == minibatch_y).item()
    return loss / data_len, accuracy / data_len
