__author__ = 'Richard Diehl Martinez, John Kamalu'


import os
import math
import datetime
import logging

import pyaml

import torch


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


def stopwatch(func):
    """
    Time a given function.
    """
    def _func(*args, **kwargs):
        print("tic...")
        avant = datetime.datetime.now()
        x = func(*args, **kwargs)
        apres = datetime.datetime.now()
        print("...toc ({0} sec.)".format(apres - avant))
        return x
    return _func
