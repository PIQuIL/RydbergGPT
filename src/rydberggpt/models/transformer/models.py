import torch
import torch.nn as nn
import torch.nn.functional as F

from rydberggpt.models.transformer.layers import DecoderLayer, EncoderLayer
from rydberggpt.models.transformer.utils import clones


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, tgt, cond):
        "Take in and process masked src and target sequences."
        return self.decode(tgt, self.encode(cond))

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, tgt, memory):
        return self.decoder(self.tgt_embed(tgt), memory)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return x
        # return self.norm(x)


class Decoder(nn.Module):
    """this is the core of the transformer which is a stack n encoder layers"""

    def __init__(self, layer, n_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        """pass the (masked) input trough all layers"""
        for layer in self.layers:
            x = layer(x, memory)
        return x
        # return self.norm(x)


class Generator(nn.Module):
    """
    linear + softmax layer for generation step. vocab_size for Rydberg is 2.
    """

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
