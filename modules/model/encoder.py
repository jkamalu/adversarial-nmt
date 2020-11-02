__author__ = 'John Kamalu'

'''
A wrapper class for BERT/vanilla self-attention models for inclusion
in NMT and sequence-to-sequence pipelines
'''

import argparse
import torch.nn as nn

from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          CamembertConfig, CamembertModel, CamembertTokenizer,
                          XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer)


MODEL_CLASSES = {
    'en': (RobertaConfig, RobertaModel, "roberta-base"),
    'fr': (CamembertConfig, CamembertModel, "camembert-base"),
    'multi': (XLMRobertaConfig, XLMRobertaModel, "xlm-roberta-base")
}


class Encoder(nn.Module):

    def __init__(self, impl):
        super().__init__()

        self.is_initialized = False
        self.impl = impl

    @classmethod
    def init_from_config(cls, impl, encoder_kwargs, language):
        module = cls(impl)
        
        if impl == "bert":
            _, model_class, weights = MODEL_CLASSES[language]
            module.model = model_class.from_pretrained(weights)
        else:
            raise NotImplementedError

        module.is_initialized = True

        return module

    def forward(self, src, lengths=None):
        assert self.is_initialized

        if self.impl == "bert":
            # huggingface models assume (B, L) and return [sequence_output, pooled_output, ...]
            output = self.model(src)[0]

        return output
