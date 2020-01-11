__author__ = 'John Kamalu'

''' Wrapper class oover the onmt TransformerDecoder'''

from onmt.decoders import TransformerDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDec(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.vocab_size = kwargs['embeddings'].make_embedding.emb_luts[0].num_embeddings
        self.hidden_size = kwargs["d_model"]
        self.max_seq_len = kwargs['maxlen']

        self.embeddings = kwargs['embeddings'].make_embedding.emb_luts[0]
        self.bridge = nn.Linear(self.max_seq_len , 1)
        #TODO: eventually we also want to incorporate an attention mechanism

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.projection = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, *args, **kwargs):
        '''
        Input is specified as either teacher forcing or not in the training loop.
        Hidden is the hidden state representation of the BERT encoder.
        '''

        emb_output = self.embeddings(input)
        context_vec = self.bridge(hidden.transpose(1,2))
        output = F.relu(output)
        gru_output, gru_hidden = self.gru(output.transpose(0,1), context_vec.permute(2,0,1))
        output = self.softmax(self.projection(gru_output[0]))

        return output

    def init_state(self, *args):
        ''' Implemented for consistency with TransformerDec'''
        pass

class TransformerDec(TransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = self.embeddings.make_embedding.emb_luts[0].num_embeddings
        self.projection = torch.nn.Linear(kwargs["d_model"], self.vocab_size)

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
        src = src.transpose(0, 1).unsqueeze(-1)
        # OpenNMT-py TransformerDecoder.init_state does not use args[1:3]
        super().init_state(src, None, None)



class Decoder(nn.Module):
    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)
        self.type = type

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def init_from_config(cls, type="GRU", *args, **kwargs):
        assert("embeddings" in kwargs), "Must specify embeddings when initializing a GRU Decoder."

        module = cls(type)
        if(type == "GRU"):
            module.model = GRUDec(*args, **kwargs)
        elif(type == "Transformer"):
            module.model = TransformerDec(*args, **kwargs)
        else:
            raise Exception("Invalid decoder type specified - specify either GRU or Transformer.")
        return module
