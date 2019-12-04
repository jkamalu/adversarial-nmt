__author__ = 'John Kamalu'

'''
A wrapper class for BERT based models for inclusion in NMT
and sequence-to-sequence pipelines
'''

import argparse

import torch

from lib.huggingface.transformers import (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                          CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)

MODEL_CLASSES = {
    'english': (RobertaConfig, RobertaForMaskedLM),
    'french': (CamembertConfig, CamembertForMaskedLM)
}

class Encoder(torch.nn.Module):

    def __init__(self, language, weights, device="cpu"):
        super().__init__()

        self.language = language
        self.weights = weights

        config_class, model_class = MODEL_CLASSES[args.language]

        self.config = config_class.from_pretrained(args.weights)

        self.model = model_class.from_pretrained(args.weights,
                                                 from_tf=bool('.ckpt' in args.weights))

        self.model.to(device)

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, required=True,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--weights", type=str, required=True,
                        help="The model checkpoint for weights initialization.")

    args = parser.parse_args()

    encoder = Encoder(args.language, args.weights)
