import os
import sys
import logging
from itertools import chain
from argparse import ArgumentParser

from transformers import RobertaTokenizer, CamembertTokenizer, BertTokenizer

import torch; torch.autograd.set_detect_anomaly(True)
import torch.multiprocessing as mp
from torch.nn import ModuleDict
from torch.utils.data import Subset
from torch.optim import Adam, SGD, RMSprop

from modules.data import EuroparlDataset
from modules.run import train, evaluate
from modules.utils import load_config, path_to_data, path_to_config, load_checkpoint
from modules.model import BidirectionalTranslator


TOKENIZERS = {
    "en": lambda: RobertaTokenizer.from_pretrained('roberta-base'),
    "fr": lambda: CamembertTokenizer.from_pretrained('camembert-base'),
    "multi": lambda: BertTokenizer.from_pretrained('bert-base-multilingual-cased')
}

DATA = {
    frozenset(["en", "fr"]): {"en":"europarl-v7.fr-en.en", "fr":"europarl-v7.fr-en.fr"}
}


def datasets(config):
    data_path = path_to_data("europarl-v7")

    if config["shared_encoder"]:
        tokenizer_l1 = TOKENIZERS[config["multi"]]()
        tokenizer_l2 = tokenizer_l1
    else:
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
    
    logging.info("creating model and optimizer")
    model = BidirectionalTranslator(config)
            
    # split generator and discriminator parameters
    gen_params = chain.from_iterable(
        map(
            lambda x: x.parameters(), 
            filter(
                lambda x: type(x) != ModuleDict, 
                model.children()
            )
        )
    )
    dis_params = model.discriminators.parameters()
    
    gen_optimizer = Adam(gen_params, **config["adam"])
    dis_optimizer = RMSprop(dis_params, **config["rmsp"])
    
    step = 0
    
    if args.iter is not None:
        logging.info("loading model and optimizer parameters")
        ckpt = load_checkpoint(model, gen_optimizer, dis_optimizer, args.iter, config["experiment"])
        model, gen_optimizer, dis_optimizer, step = ckpt

    logging.info("starting the {} routine from step {}.".format(config["mode"], step))
    if config["dist"]:
        raise NotImplementedError
    else:
        if config["mode"] == "train":
            train(model, gen_optimizer, dis_optimizer, dataset_train, dataset_valid, step, config)
        else:
            evaluate(model, dataset_valid, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--experiment", "-e", type=str, default=path_to_config("bert-vanilla-hidden.yml"))
    parser.add_argument("--iter", "-i", type=str, default=None)
    parser.add_argument("--mode", "-m", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--dist", "-d", action="store_true")
    parser.add_argument("--shared_encoder", "-s", action="store_true")
    
    args = parser.parse_args()
    
    main(args)