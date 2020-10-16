__author__ = 'John Kamalu'

import random
from argparse import ArgumentParser, Namespace

from fairseq.models.transformer import TransformerDecoder as FairseqDecoder
from fairseq.models.transformer import TransformerModel as FairseqModel
from fairseq.models.transformer import base_architecture
from fairseq.models.fairseq_encoder import EncoderOut

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GRUDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.vocab_size = kwargs['embeddings'].make_embedding.emb_luts[0].num_embeddings
        self.hidden_size = kwargs["d_model"]
        self.max_seq_len = kwargs['maxlen']

        self.embeddings = kwargs['embeddings'].make_embedding.emb_luts[0]
        self.bridge = nn.Linear(self.max_seq_len , 1)
        self.step_func = self.step_basic
        if "attention_mech" in kwargs:
            self.step_func = self.step_bahdanau if kwargs["attention_mech"] == "bahdanau_attention" \
                            else self.step_basic

        self.teacher_forcing_prob = 1
        if "teacher_forcing_prob" in kwargs:
            self.teacher_forcing_prob = kwargs["teacher_forcing_prob"]

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.projection = nn.Linear(self.hidden_size, self.vocab_size)

    def step_basic(self, input, hidden, *args, **kwargs):
        '''
        Defines one step through the GRU to predict the next input.
        '''
        emb_output = F.relu(self.embeddings(input))
        initial_state = self.bridge(hidden.transpose(1,2))

        gru_output, _ = self.gru(emb_output.transpose(0,1), initial_state.permute(2,0,1))
        output = self.projection(gru_output[0]) # Loss uses logits directly
        return output

    def step_bahdanau(self, input, hidden, *args, **kwargs):
        '''
        Defines one step through the GRU to predict the next input. Uses
        standard bahdanau attention for decoding.
        '''

        emb_output = F.relu(self.embeddings(input))
        initial_state = self.bridge(hidden.transpose(1,2))

        #Bahdanau attenntion
        alignment_scores = F.softmax(torch.matmul(emb_output, hidden.transpose(1,2)), dim=-1)
        context_vecs = torch.matmul(alignment_scores, hidden)

        gru_output, _ = self.gru(context_vecs.transpose(0,1), initial_state.permute(2,0,1))
        output = self.projection(gru_output[0]) # Loss uses logits directly
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
                curr_prediction = self.step_func(input[:, :i+1], hidden, *args, **kwargs)
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
                curr_prediction = self.step_func(curr_input, hidden, *args, **kwargs)
                predictions.append(curr_prediction)

        output = torch.stack(predictions, dim=1)
        return output

    def init_state(self, *args):
        ''' Implemented for consistency with TransformerDec'''
        pass
