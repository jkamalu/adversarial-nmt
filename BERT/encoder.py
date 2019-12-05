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
        self.linear_classify = torch.nn.Linear(768, 1)
        self.model.to(device)

    def forward(self, x):
        encoder_out = self.model(x)
        assert(len(encoder_out) == 2), "Output of the encoder needs to have pooled layer in second dimension"
        language_classification_out = self.linear_classify(encoder_out[1])
        return (encoder_out, language_classification_out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, required=True,
                        help="The model architecture to be fine-tuned.")

    args = parser.parse_args()

    encoder = Encoder(args.language, args.weights)
