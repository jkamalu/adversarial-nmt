import os
import sys
import logging
from argparse import ArgumentParser

from transformers import RobertaTokenizer, CamembertTokenizer

import torch; torch.autograd.set_detect_anomaly(True)
import torch.multiprocessing as mp
from torch.utils.data import Subset

from modules.data import EuroparlDataset
from modules.run import train, evaluate
from modules.utils import load_config, path_to_data, path_to_config
from modules.model import BidirectionalTranslator


TOKENIZERS = {
    "en": lambda: RobertaTokenizer.from_pretrained('roberta-base'),
    "fr": lambda: CamembertTokenizer.from_pretrained('camembert-base')
}

DATA = {
    frozenset(["en", "fr"]): {"en":"europarl-v7.fr-en.en", "fr":"europarl-v7.fr-en.fr"}
}


def datasets(config):
    data_path = path_to_data("europarl-v7")
    
    tokenizer_l1 = TOKENIZERS[config["l1"]]()
    tokenizer_l2 = TOKENIZERS[config["l2"]]()
    
    pair = frozenset([config["l1"], config["l2"]])
    
    dataset = EuroparlDataset.to_hdf5(
        data_path,
        os.path.join(data_path, DATA[pair][config["l1"]]),
        os.path.join(data_path, DATA[pair][config["l2"]]),
        tokenizer_l1,
        tokenizer_l2,
        config["minlen"],
        config["maxlen"]
    )
    
    dataset_train, dataset_valid = torch.utils.data.random_split(
        dataset,
        [len(dataset) - config["n_valid"], config["n_valid"]],
        generator=torch.Generator().manual_seed(config["seed"])
    )
    
    # monkey patch torch.utils.data.Subset
    Subset.tokenizer_l1 = dataset.tokenizer_l1
    Subset.tokenizer_l2 = dataset.tokenizer_l2
    
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