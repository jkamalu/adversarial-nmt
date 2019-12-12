__author__ = 'John Kamalu'

''' Wrapper class oover the onmt TransformerDecoder'''

from onmt.decoders import TransformerDecoder
import torch

class Decoder(TransformerDecoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v = self.embeddings.make_embedding.emb_luts[0].num_embeddings
        self.projection = torch.nn.Linear(kwargs["d_model"], self.v)

    def forward(self, tgt, memory_bank, memory_lengths, step=None, **kwargs):
        # OpenNMT-py TransformerDecoder *FOOLISHLY* assume (L, B, D)
        tgt = tgt.transpose(0, 1).unsqueeze(-1)
        memory_bank = memory_bank.transpose(0, 1)
        # OpenNMT-py TransformerEncoder returns out_x, attn_x
        dec_outs, attns = super().forward(tgt, memory_bank, memory_lengths=memory_lengths, step=step, **kwargs)
        # Project to the vocabulary dimension and enforce (B, L, D)
        dec_outs = dec_outs.transpose(0, 1)
        dec_outs = self.projection(dec_outs)
        return dec_outs
    
    def init_state(self, src):
        # OpenNMT-py TransformerDecoder *FOOLISHLY* assume (L, B, D)
        src = src.transpose(0,1).unsqueeze(-1)
        # OpenNMT-py TransformerDecoder.init_state does not use args[1:3]
        super().init_state(src, None, None)
