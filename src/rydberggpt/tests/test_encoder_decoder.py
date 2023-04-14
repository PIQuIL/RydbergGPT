import copy
from dataclasses import dataclass

import pytest
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from rydberggpt.data.loading.dataset_rydberg import get_dataloaders, load_dataset
from rydberggpt.models import TransformerWavefunction
from rydberggpt.models.transformer.layers import DecoderLayer, EncoderLayer
from rydberggpt.models.transformer.loss import LabelSmoothing
from rydberggpt.models.transformer.models import (
    Decoder,
    Encoder,
    EncoderDecoder,
    Generator,
)
from rydberggpt.models.transformer.modules import PositionwiseFeedForward

# seed everything
torch.manual_seed(0)


def init_data():
    # seed everything
    torch.manual_seed(0)

    # LOAD MODEL
    @dataclass
    class Config:
        # transformer
        num_heads: int = 8
        d_model: int = 32
        num_blocks: int = 2
        d_ff = 4 * d_model
        dropout = 0.0
        # training
        num_epochs: int = 2
        batch_size: int = 16
        learning_rate: float = 0.01
        # dataset
        num_atoms: int = None
        num_samples: int = None
        delta: float = None

    # LOAD DATA
    data, dataset_config = load_dataset(delta_id=0)

    config = Config(
        num_atoms=dataset_config.num_atoms,
        num_samples=dataset_config.num_samples,
        delta=dataset_config.delta,
    )

    train_loader, val_loader, test_loader = get_dataloaders(data, config)

    return config, train_loader, val_loader, test_loader


def init_model(config):
    c = copy.deepcopy
    attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
    ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)

    model = TransformerWavefunction(
        encoder=Encoder(
            EncoderLayer(config.d_model, c(attn), c(ff), config.dropout),
            config.num_blocks,
        ),
        decoder=Decoder(
            DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout),
            config.num_blocks,
        ),
        src_embed=nn.Linear(4, config.d_model),
        tgt_embed=nn.Linear(2, config.d_model),
        generator=Generator(config.d_model, 2),
        config=config,
    )
    return model


def test_encoder_decoder(model):
    # define the loss function
    criterion = LabelSmoothing(0.0)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # H = torch.rand((config.batch_size, dataset_config.num_atoms, 4), dtype=torch.float)

    # loop over the data for the specified number of epochs
    for i in range(50):
        # set the model to train mode
        model.train()

        # loop over the training data in batches
        for i, batch in enumerate(train_loader):
            inputs, condition = batch
            inputs = nn.functional.one_hot(inputs, 2)
            inputs = inputs.to(torch.float)

            # print(condition.shape)
            optimizer.zero_grad()
            # assert inputs.dtype == torch.int64

            temp = model.forward(inputs, condition)
            log_cond_probs = model.generator(temp)
            loss = criterion(log_cond_probs, inputs)
            print(loss)
            # assert loss is not Nan
            assert not torch.isnan(loss), "Loss is NaN"
            loss.backward()
            optimizer.step()


def test_encoder_decoder_sequentially(model):
    c = copy.deepcopy
    attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
    ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)

    # define the loss function
    criterion = LabelSmoothing(0.0)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # H = torch.rand((config.batch_size, dataset_config.num_atoms, 4), dtype=torch.float)

    import numpy as np

    # loop over the data for the specified number of epochs
    for i in range(50):
        # for epoch in range(config.num_epochs):
        # set the model to train mode
        model.train()

        # loop over the training data in batches
        for i, batch in enumerate(train_loader):
            inputs, condition = batch
            inputs = nn.functional.one_hot(inputs, 2)
            inputs = inputs.to(torch.float)

            # print(condition.shape)
            optimizer.zero_grad()
            # assert inputs.dtype == torch.int64

            # 1. embed the condition or prompt
            src_emb = nn.Linear(4, config.d_model)(condition)

            # 2. run the encoder
            encoder_out = Encoder(
                EncoderLayer(config.d_model, c(attn), c(ff), config.dropout),
                config.num_blocks,
            )(src_emb)
            assert encoder_out.shape == (
                config.batch_size,
                config.num_atoms,
                config.d_model,
            )

            # 3. embed the inputs
            dec_emb = nn.Linear(2, config.d_model)(inputs)
            assert dec_emb.shape == (
                config.batch_size,
                config.num_atoms,
                config.d_model,
            )

            # 4. run the decoder
            decoder_out = Decoder(
                DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout),
                config.num_blocks,
            )(dec_emb, encoder_out)

            # 5. generate the output log cond probabilities
            log_cond_probs = Generator(config.d_model, 2)(decoder_out)

            # temp = model.forward(inputs, condition)
            # log_cond_probs = model.generator(temp)
            loss = criterion(log_cond_probs, inputs)

            assert not torch.isnan(loss), "Loss is NaN"
            # check that loss is going minized
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    config, train_loader, val_loader, test_loader = init_data()
    model = init_model(config)
    test_encoder_decoder(model)
    test_encoder_decoder_sequentially(model)
