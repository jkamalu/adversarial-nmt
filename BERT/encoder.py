__author__ = 'John Kamalu'

'''
A wrapper class for BERT based models for inclusion in NMT
and sequence-to-sequence pipelines
'''

import argparse

import torch

from lib.huggingface.transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                                          CamembertConfig, CamembertModel, CamembertTokenizer)

MODEL_CLASSES = {
    'english': (RobertaConfig, RobertaModel, "roberta-base"),
    'french': (CamembertConfig, CamembertModel, "camembert-base")
}

class Encoder(torch.nn.Module):

    def __init__(self, language, device="cpu"):
        super().__init__()

        self.language = language
        _, model_class, weights = MODEL_CLASSES[language]

        self.model = model_class.from_pretrained(weights)
        self.model.to(device)

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, required=True,
                        help="The model architecture to be fine-tuned.")

    args = parser.parse_args()

    encoder = Encoder(args.language, args.weights)
