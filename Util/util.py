import logging

def print_msg(msg):
    print(msg)
    logging.info(msg)
    return

def log(epoch, log_dict, wandb):
    msg = f"epoch {epoch} -- "
    msg += ',  '.join([f"{key}: {val:.3f}" for key,val in zip(log_dict.keys(), log_dict.valuees())])
    print(msg)
    logging.info(msg)
    log_dict["epoch"] = epoch
    wandb.log(log_dict)