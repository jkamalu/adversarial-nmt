__author__ = 'John Kamalu'

''' Wrapper class oover the onmt TransformerDecoder'''

from onmt.decoders import TransformerDecoder
import random
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

        self.teacher_forcing_prob = 1
        if ("teacher_forcing_prob" in kwargs):
            self.teacher_forcing_prob = kwargs["teacher_forcing_prob"]

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.projection = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def step(self, input, hidden, *args, **kwargs):
        '''
        Defines one step through the LSTM to predict the next input.
        '''
        emb_output = self.embeddings(input)
        context_vec = self.bridge(hidden.transpose(1,2))
        output = F.relu(emb_output)
        gru_output, _ = self.gru(output.transpose(0,1), context_vec.permute(2,0,1))
        output = self.softmax(self.projection(gru_output[0]))
        return output


    def forward(self, input, hidden, *args, **kwargs):
        '''
        Input is specified as either teacher forcing or not in the training loop.
        Hidden is the hidden state representation of the BERT encoder.

        We specify a certain probability that we use either teacher forcing or
        previous predictions - this probability is passed in as a parameter and
        set to self.teacher_forcing_prob during initialization of the instance.
        '''

        using_teacher_forcing = True if random.random() < self.teacher_forcing_prob else False

        predictions = []

        if using_teacher_forcing:
            for i in range(self.max_seq_len - 1):
                curr_prediction = self.step(input[:, :i+1], hidden, *args, **kwargs)
                predictions.append(curr_prediction)
        else:
            sos_token = input[:, 0].unsqueeze(1)
            for i in range(self.max_seq_len - 1):
                # curr_input variable used to keep track of current input to the LSTM model
                if i == 0:
                    curr_input = sos_token
                else:
                    argmax_predictions = torch.argmax(torch.stack(predictions, dim=1), dim=2)
                    curr_input = torch.cat([sos_token, argmax_predictions], dim=1)
                curr_prediction = self.step(curr_input, hidden, *args, **kwargs)
                predictions.append(curr_prediction)

        output = torch.stack(predictions, dim=1)
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
