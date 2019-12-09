__author__ = 'John Kamalu'

'''
A wrapper class for BERT/vanilla Transformer models for inclusion
in NMT and sequence-to-sequence pipelines
'''

import argparse
import torch

from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          CamembertConfig, CamembertModel, CamembertTokenizer)

MODEL_CLASSES = {
    'english': (RobertaConfig, RobertaModel, "roberta-base"),
    'french': (CamembertConfig, CamembertModel, "camembert-base")
}

# imports to enable TransformerEncoder to work
import torch.nn as nn
from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask

# Copied from ONMT
class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

# Copied from ONMT
class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()

        # OpenNMT-py *FOOLISHLY* assumes (L, B, D)
        # Force the mask to be size max_len i.e. L
        mask = ~sequence_mask(lengths, max_len=src.shape[0]).unsqueeze(1)

        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return out.contiguous()

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)

# Encoder wrapper
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
            encoder_out = self.model(x, lengths=lengths)
#             encoder_out = encoder_out.transpose(0, 1)

        return encoder_out
