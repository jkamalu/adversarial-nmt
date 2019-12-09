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
    
    def __init__(self, use_bert, **kwargs):
        super().__init__(**kwargs)
        
        self.is_initialized = False
        self.use_bert = use_bert

    @classmethod
    def init_bert(cls, language):
        module = cls(use_bert=True)
        
        _, model_class, weights = MODEL_CLASSES[language]

        module.model = model_class.from_pretrained(weights)
        module.embeddings = module.model.get_input_embeddings()
        
        module.is_initialized = True
        
        return module
    
    @classmethod
    def init_vanilla(cls, **kwargs):
        module = cls(use_bert=False)
        
        module.model = TransformerEncoder(**kwargs)
        module.embeddings = kwargs["embeddings"]
        
        module.is_initialized = True
        
        return module
        

    def forward(self, x, lengths=None):
        
        assert (self.is_initialized), "Encoder.init_bert or Encoder.init_vanilla must be called before the module can be used."
        
        if not self.use_bert and lengths is None:
            raise ValueError("OpenNMT-py TransformerEncoder requires lengths to be not None.") 
        
        if self.use_bert:
            # huggingface models assume (B, L)
            encoder_out = self.model(x)
            # huggingface models return [sequence_output, pooled_output, ...]
            return encoder_out[1]
        else:
            # OpenNMT-py TransformerEncoder *FOOLISHLY* assume (L, B, D)
            x = x.transpose(0, 1).unsqueeze(-1)
            # OpenNMT-py TransformerEncoder returns emb_x, out_x, len_x
            _, encoder_out, _  = self.model(x, lengths=lengths)
            encoder_out = encoder_out.transpose(0, 1)
        
        return encoder_out
