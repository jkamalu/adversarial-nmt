__author__ = 'John Kamalu'

''' Wrapper class oover the onmt TransformerDecoder'''

from onmt.decoders import TransformerDecoder
import torch

class Decoder(TransformerDecoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        dec_outs, attns = super().forward(tgt, memory_bank, step=step, **kwargs)

        weight = self.embeddings.make_embedding.emb_luts[0].weight

        dec_outs = torch.einsum('lbd,vd->lbv', dec_outs, weight).transpose(0,1)

        return dec_outs, attns
