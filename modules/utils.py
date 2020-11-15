import os
import math
import datetime
import logging
from itertools import chain

import pyaml

import torch
from torch.nn import ModuleDict
from torch.optim import Adam, SGD, RMSprop


def path_to_config(name):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", name)


def path_to_output(name):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", name)


def path_to_data(directory):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", directory)


def init_output_dirs(experiment):
    directory = path_to_output(experiment)

    ckpt_dir = "{}/checkpoints".format(directory)
    runs_dir = "{}/tensorboard".format(directory)

    try: 
        os.makedirs(ckpt_dir)
        os.makedirs(runs_dir)
    except FileExistsError: 
        logging.warning("ckpt and log dirs already exist for {}".format(experiment))
        
    return ckpt_dir, runs_dir


def load_config(path):
    """
    Load the config file and make any dynamic edits.
    """
    with open(path, "rt") as reader:
        config = pyaml.yaml.load(reader, Loader=pyaml.yaml.Loader)
    if config["regularization"]["type"] is None or config["regularization"]["type"] == [None]:
        config["regularization"]["type"] = []
    if "attention" in config["regularization"]["type"]:
        raise NotImplementedError
    
    config["experiment"] = os.path.splitext(os.path.basename(path))[0]
    config["ckpt_dir"], config["runs_dir"] = init_output_dirs(config["experiment"])
        
    return config


def build_optimizers(model, gen_opt_kwargs, dis_opt_kwargs):
    gen_optimizer = Adam(model.generator.parameters(), **gen_opt_kwargs)
    if dis_opt_kwargs:
        dis_optimizer = RMSprop(model.discriminators.parameters(), **dis_opt_kwargs)
        return gen_optimizer, dis_optimizer
    else:
        return gen_optimizer, None


def save_checkpoint(model, gen_optimizer, dis_optimizer, step, experiment):
    ckpt = "{0}.pt".format(str(step).zfill(6))

    state_dict = {}
    state_dict["model_state_dict"] = model.state_dict()
    state_dict["gen_optimizer_state_dict"] = gen_optimizer.state_dict()
    state_dict["step"] = step

    if dis_optimizer is not None:
        state_dict["dis_optimizer_state_dict"] = dis_optimizer.state_dict()

    torch.save(state_dict, os.path.join(path_to_output(experiment), "checkpoints", ckpt))


def load_checkpoint(model, gen_opt_kwargs, dis_opt_kwargs, step, experiment):
    ckpt = "{0}.pt".format(str(step).zfill(6))
    state_dict = torch.load(os.path.join(path_to_output(experiment), "checkpoints", ckpt))

    model.load_state_dict(state_dict["model_state_dict"])

    gen_optimizer, dis_optimizer = build_optimizers(model, gen_opt_kwargs, dis_opt_kwargs)

    gen_optimizer.load_state_dict(state_dict["gen_optimizer_state_dict"])
    for state in gen_optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if dis_optimizer is not None:
        dis_optimizer.load_state_dict(state_dict["dis_optimizer_state_dict"])
        for state in dis_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    return model, gen_optimizer, dis_optimizer, state_dict["step"]
