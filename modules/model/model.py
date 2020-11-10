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
        self.encoder_en = Encoder.init_from_config("bert", config["encoder_kwargs"]["bert"], "english")
        self.encoder_fr = Encoder.init_from_config("bert", config["encoder_kwargs"]["bert"], "french")
        
        # Embeddings
        if enc == "bert":
            embeddings_en = self.encoder_en.model.get_input_embeddings()
            embeddings_fr = self.encoder_fr.model.get_input_embeddings()
        else:
            raise NotImplementedError
            
        # Decoders
        dec = config["decoder"]
        self.decoder_en = Decoder.init_from_config(dec, config["decoder_kwargs"][dec], embeddings_en)
        self.decoder_fr = Decoder.init_from_config(dec, config["decoder_kwargs"][dec], embeddings_fr)
        
        # Discriminators
        self.discriminators = nn.ModuleDict({
            regularization: Discriminator(regularization, config["discriminator_kwargs"])
                for regularization in config["regularization"]["type"]
        })

    def forward(self, sents_en, sents_no_eos_en, lengths_en, sents_fr, sents_no_eos_fr, lengths_fr):
        # English to French
        enc_out_en = self.encoder_en(sents_en, lengths=lengths_en)
        dec_out_fr = self.decoder_fr(sents_no_eos_fr, enc_out_en, lengths_en)

        # French to English
        enc_out_fr = self.encoder_fr(sents_fr, lengths=lengths_fr)
        dec_out_en = self.decoder_en(sents_no_eos_en, enc_out_fr, lengths_fr)
        
        # English to French encoder weights
        en_fr_enc_w = self.encoder_en.state_dict()

        # French to English encoder weights
        fr_en_enc_w = self.encoder_fr.state_dict()

        return sents_en, sents_fr, enc_out_en, enc_out_fr, dec_out_en, dec_out_fr, en_fr_enc_w, fr_en_enc_w

    def discriminate(self, regularization, enc_out):
        return self.discriminators[regularization](enc_out)