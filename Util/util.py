import logging
from copy import deepcopy

def print_msg(msg):
    print(msg)
    logging.info(msg)
    return

def log(epoch, log_dict, wandb):
    msg = f"epoch {epoch} -- "
    msg += ',  '.join([f"{key}: {val:.3f}" for key,val in zip(log_dict.keys(), log_dict.values())])
    print(msg)
    logging.info(msg)
    log_dict["epoch"] = epoch
    wandb.log(log_dict)

def log_task(task_id, log_dict, wandb):
    msg = f"After task {task_id} -- "
    msg += ',  '.join([f"{key}: {val:.3f}" for key,val in zip(log_dict.keys(), log_dict.values())])
    print(msg)
    logging.info(msg)
    log_dict["task_id"] = task_id
    wandb.log(log_dict)

def copy_freeze(model):
    model_copy = deepcopy(model)
    for param in model_copy.parameters():
        param.requires_grad = False
    return model_copy