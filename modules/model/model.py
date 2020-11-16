import torch
import torch.nn as nn

from modules.model import Encoder
from modules.model import Decoder
from modules.model import Discriminator

class BidirectionalTranslator(nn.Module):
    
    def __init__(self, config):
        """
        Initialize the bidirectional translator model.
        
        We always load in the BERT models to access the pretrained
        embeddings. We tie the embedding layer and decoder output layer.
        """
        super().__init__()
        
        # Encoders
        enc = config["encoder"]
        encoder = Encoder.init_from_config("bert", config["encoder_kwargs"]["bert"], "multi")
        
        # Embeddings
        if enc == "bert":
            embeddings = encoder.model.get_input_embeddings()
        else:
            raise NotImplementedError
            
        # Decoders
        dec = config["decoder"]
        decoder_l1 = Decoder.init_from_config(dec, config["decoder_kwargs"][dec], embeddings)
        decoder_l2 = Decoder.init_from_config(dec, config["decoder_kwargs"][dec], embeddings)

        # Generator
        self.generator = nn.ModuleDict({
            "encoder": encoder,
            "decoder_l1": decoder_l1,
            "decoder_l2": decoder_l2
        })
        
        # Discriminators
        self.discriminators = nn.ModuleDict({
            regularization: Discriminator(regularization, config["discriminator_kwargs"])
                for regularization in config["regularization"]["type"]
        })

    def forward(self, sents_l1, sents_no_eos_l1, lengths_l1, sents_l2, sents_no_eos_l2, lengths_l2):
        # L1 to L2
        enc_out_l1 = self.generator.encoder(sents_l1, lengths=lengths_l1)
        dec_out_l2 = self.generator.decoder_l2(sents_no_eos_l2, enc_out_l1, lengths_l1)

        # L2 to L1
        enc_out_l2 = self.generator.encoder(sents_l2, lengths=lengths_l2)
        dec_out_l1 = self.generator.decoder_l1(sents_no_eos_l1, enc_out_l2, lengths_l2)
        
        return sents_l1, sents_l2, enc_out_l1, enc_out_l2, dec_out_l1, dec_out_l2

    def discriminate(self, regularization, enc_out):
        return self.discriminators[regularization](enc_out)
