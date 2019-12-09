__author__ = 'John Kamalu'

'''
A wrapper class for BERT/vanilla Transformer models for inclusion 
in NMT and sequence-to-sequence pipelines
'''

import argparse
import torch
from onmt.encoders import TransformerEncoder
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          CamembertConfig, CamembertModel, CamembertTokenizer)

MODEL_CLASSES = {
    'english': (RobertaConfig, RobertaModel, "roberta-base"),
    'french': (CamembertConfig, CamembertModel, "camembert-base")
}

class Encoder(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.is_initialized = False

    @classmethod
    def init_bert(cls, language):
        module = cls()
        
        _, model_class, weights = MODEL_CLASSES[language]

        module.model = model_class.from_pretrained(weights)
        
        module.is_initialized = True
        
        return module
    
    @classmethod
    def init_vanilla(cls, **kwargs):
        module = cls()
        
        module.model = TransformerEncoder(**kwargs)
        
        return module
        

    def forward(self, x):
        
        assert (self.is_initialized), "Encoder.init_bert or Encoder.init_vanilla must be called."
        
        encoder_out = self.model(x)
        assert(len(encoder_out) == 2), "Output of the encoder needs to have pooled layer in second dimension"
        
        return encoder_out

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, required=True,
                        help="The model architecture to be fine-tuned.")

    args = parser.parse_args()

    encoder = Encoder(args.language, args.weights)
