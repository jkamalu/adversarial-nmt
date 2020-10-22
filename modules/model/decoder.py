__author__ = 'John Kamalu'

from argparse import ArgumentParser, Namespace

from fairseq.models.transformer import TransformerDecoder as FairseqDecoder
from fairseq.models.transformer import TransformerModel as FairseqModel
from fairseq.models.transformer import base_architecture
from fairseq.models.fairseq_encoder import EncoderOut

import torch
import torch.nn as nn


class Decoder(nn.Module):
    
    def __init__(self, impl):
        super().__init__()

        self.is_initialized = False
        self.impl = impl
    
    @classmethod
    def init_from_config(cls, impl, decoder_kwargs, embedding):

        module = cls(impl)
        
        module.embedding = embedding
        module.decoder_kwargs = decoder_kwargs
        
        if impl == "fairseq":
            args = {}
            
            # fairseq default args
            ap = ArgumentParser()
            FairseqModel.add_args(ap)
            args.update(vars(ap.parse_args("")))
            
            # fairseq base architecture args
            ns = Namespace(**decoder_kwargs)
            base_architecture(ns)
            args.update(vars(ns))
            
            # our args
            args.update(decoder_kwargs)
            
            namespace = Namespace(**args)
            dumb_dict = {0 for _ in range(embedding.weight.shape[0])}
            
            module.model = FairseqDecoder(namespace, dumb_dict, embedding)
        else:
            raise NotImplementedError()
            
        module.is_initialized = True
            
        return module

    def forward(self, tgt, enc_out, src_len):
        assert self.is_initialized
        
        if self.impl == "fairseq":
            
            B, L, H = enc_out.shape
                        
            encoder_out = EncoderOut(
                enc_out.transpose(0, 1), 
                torch.arange(L, device=src_len.device).unsqueeze(0).expand((B, L)) - src_len.unsqueeze(1) >= 1,
                None, 
                None, 
                None, 
                None
            )
            output, _ = self.model.forward(tgt, encoder_out=encoder_out, src_lengths=src_len)
        
        return output
