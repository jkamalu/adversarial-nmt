from fairseq.models.transformer import Embedding as init_embedding

import torch


class Embedding(torch.nn.Embedding)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_pretrained = False
    
    @classmethod
    def from_pretrained(cls, torch_embeddings):
        params = {
            "word_padding_idx": torch_embeddings.padding_idx,
            "word_vocab_size": torch_embeddings.num_embeddings,
            "word_vec_size": torch_embeddings.embedding_dim,
            "position_encoding": True
        }
        module = cls(**params)
        module.word_lut.weight.data.copy_(torch_embeddings.weight)
        module.is_pretrained = True
        return module
        