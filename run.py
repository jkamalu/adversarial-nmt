import os
import sys
import logging
from argparse import ArgumentParser

from transformers import RobertaTokenizer, CamembertTokenizer

import torch
import torch.multiprocessing as mp

from modules.data import TextDataset
from modules.run import train, evaluate
from modules.utils import load_config, path_to_data, path_to_config
from modules.model import BidirectionalTranslator


def datasets(config):
    data_path = path_to_data("europarl-v7")
    
    tokenizer_en = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer_fr = CamembertTokenizer.from_pretrained('camembert-base')
    
    dataset_train = None
    if config["mode"] == "train":
        dataset_train = TextDataset(
            data_path, 
            tokenizer_en, 
            tokenizer_fr, 
            training=True, 
            minlen=config["minlen"],
            maxlen=config["maxlen"]
        )

    dataset_valid = TextDataset(
        data_path,
        tokenizer_en, 
        tokenizer_fr, 
        training=False, 
        minlen=config["minlen"],
        maxlen=config["maxlen"]
    )
    
    return dataset_train, dataset_valid

    
def main(args):
    torch.manual_seed(0)
    
    config = {k:v for k, v in args._get_kwargs()}
    config.update(load_config(path_to_config(config["experiment"])))

    logging.info("creating europarl-v7 datasets.")    
    dataset_train, dataset_valid = datasets(config)
    
    logging.info("creating bidirectional translator.")
    model = BidirectionalTranslator(config)
    
    logging.info("starting the {} routine.".format(config["mode"]))
    if config["dist"]:
        raise NotImplementedError("Must redesign data pipeline before distributed data parallel training is feasible.")
    else:
        if config["mode"] == "train":
            train(model, dataset_train, dataset_valid, config)
        else:
            evaluate(model, dataset_valid, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--experiment", "-e", type=str, default=path_to_config("bert-vanilla-hidden.yml"))
    parser.add_argument("--mode", "-m", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--dist", "-d", action="store_true")
    
    args = parser.parse_args()
    
    main(args)