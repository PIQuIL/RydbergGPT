import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from rydberggpt.models.transformer.utils import clones


class EncoderDecoder(pl.LightningModule):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module,
    ):
        """
        Initialize the EncoderDecoder class.

        Args:
            encoder (nn.Module): The encoder module.
            decoder (nn.Module): The decoder module.
            tgt_embed (nn.Module): The target embedding module.
            generator (nn.Module): The generator module.
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, tgt: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        Take in and process masked src and target sequences.

        Args:
            tgt (torch.Tensor): The target tensor of shape (batch_size, tgt_seq_length, d_model_tgt).
            src (torch.Tensor): The source tensor of shape (batch_size, src_seq_length, d_model_src).

        Returns:
            torch.Tensor: The output tensor after passing through the encoder-decoder architecture,
                          with shape (batch_size, tgt_seq_length, d_model).
        """

        memory, batch_mask = self.encode(src)

        return self.decode(tgt, memory, batch_mask)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode the source tensor.

        Args:
            src (torch.Tensor): The source tensor of shape (batch_size, src_seq_length, d_model_src).

        Returns:
            torch.Tensor: The encoded tensor of shape (batch_size, src_seq_length, d_model_tgt).
        """

        x, batch_mask = self.src_embed(src)

        return self.encoder(x, batch_mask=batch_mask), batch_mask

    def decode(
        self, tgt: torch.Tensor, memory: torch.Tensor, batch_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode the target tensor using the memory tensor.

        Args:
            tgt (torch.Tensor): The target tensor of shape (batch_size, tgt_seq_length, d_model_tgt).
            memory (torch.Tensor): The memory tensor of shape (batch_size, src_seq_length, d_model).

        Returns:
            torch.Tensor: The decoded tensor of shape (batch_size, tgt_seq_length, d_model).
        """
        return self.decoder(self.tgt_embed(tgt), memory, batch_mask=batch_mask)


class Encoder(nn.Module):
    """
    The core encoder, which consists of a stack of N layers.
    """

    def __init__(self, layer: nn.Module, N: int):
        """
        Initialize the Encoder class.

        Args:
            layer (nn.Module): A single instance of the encoder layer to be cloned.
            N (int): The number of encoder layers in the stack.
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through each layer in turn.

        Args:
            x (torch.Tensor): The input tensor to the encoder of shape (batch_size, seq_length, d_model).
            batch_mask (torch.Tensor): The mask tensor for batches.

        Returns:
            torch.Tensor: The output tensor after passing through all layers of the encoder,
                          with the same shape as the input tensor (batch_size, seq_length, d_model).
        """
        for layer in self.layers:
            x = layer(x, batch_mask=batch_mask)
        return self.norm(x)  # [batch_size, seq_length, d_model]


class Decoder(nn.Module):
    """
    The core of the transformer, which consists of a stack of decoder layers.
    """

    def __init__(self, layer: nn.Module, n_layers: int):
        """
        Initialize the Decoder class.

        Args:
            layer (nn.Module): A single instance of the decoder layer to be cloned.
            n_layers (int): The number of decoder layers in the stack.
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, batch_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pass the (masked) input through all layers of the decoder.

        Args:
            x (torch.Tensor): The input tensor to the decoder of shape (batch_size, seq_length, d_model).
            memory (torch.Tensor): The memory tensor, typically the output of the encoder.
            batch_mask (torch.Tensor): The mask tensor for batches.

        Returns:
            torch.Tensor: The output tensor after passing through all layers of the decoder of shape (batch_size, seq_length, d_model).
        """
        for layer in self.layers:
            x = layer(x, memory, batch_mask=batch_mask)
        return self.norm(x)  # [batch_size, seq_len, d_model]


class Generator(nn.Module):
    """
    Linear + softmax layer for generation step. vocab_size for Rydberg is 2.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the Generator class.

        Args:
            d_model (int): The dimension of the input features (i.e., the last dimension of the input tensor).
            vocab_size (int): The size of the vocabulary, which determines the last dimension of the output tensor.
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # [batch_size, seq_len, vocab_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the Generator.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_length, vocab_size),
                          with log-softmax applied along the last dimension.
        """

        proj_offset = self.proj(x) + 1e-10
        # taking the log prob from a almost zero prob is challenging we need to add
        return F.log_softmax(proj_offset, dim=-1)  # [batch_size, seq_len, vocab_size]
