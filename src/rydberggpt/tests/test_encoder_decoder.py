from rydberggpt.training.loss import NLLLoss

# import copy
# from dataclasses import dataclass

# import numpy as np
# import pytest
# import torch
# import torch.optim as optim
# from torch import nn
# from tqdm import tqdm

# from _bin.dataset_rydberg import get_dataloaders, load_dataset
# from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_transformer
# from rydberggpt.models.transformer.layers import DecoderLayer, EncoderLayer
# from rydberggpt.models.transformer.models import (
#     Decoder,
#     Encoder,
#     EncoderDecoder,
#     Generator,
# )
# from rydberggpt.models.transformer.modules import PositionwiseFeedForward

# # seed everything
# torch.manual_seed(0)


# def init_data():
#     # seed everything
#     torch.manual_seed(0)

#     # LOAD MODEL
#     @dataclass
#     class Config:
#         # transformer
#         num_heads: int = 8
#         d_model: int = 32
#         num_blocks: int = 2
#         d_ff = 4 * d_model
#         dropout = 0.0
#         # training
#         num_epochs: int = 2
#         batch_size: int = 16
#         learning_rate: float = 0.01
#         # dataset
#         num_atoms: int = None
#         num_samples: int = None
#         delta: float = None
#         # rydberg
#         num_states: int = 2
#         num_encoder_embedding_dims: int = 4

#     # LOAD DATA
#     data, dataset_config = load_dataset(delta_id=0)

#     config = Config(
#         num_atoms=dataset_config.num_atoms,
#         num_samples=dataset_config.num_samples,
#         delta=dataset_config.delta,
#     )

#     train_loader, val_loader, test_loader = get_dataloaders(data, config)

#     return config, train_loader, val_loader, test_loader


# def test_encoder_decoder():
#     config, train_loader, val_loader, test_loader = init_data()
#     model = get_rydberg_transformer(config)
#     # define the loss function
#     criterion = LabelSmoothing()
#     optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

#     # loop over the data for the specified number of epochs
#     for i in range(10):
#         # set the model to train mode
#         model.train()

#         # loop over the training data in batches
#         for i, batch in enumerate(train_loader):
#             inputs, condition = batch
#             inputs = nn.functional.one_hot(inputs, 2)
#             inputs = inputs.to(torch.float)
#             optimizer.zero_grad()

#             embedding = model.forward(inputs, condition)
#             log_cond_probs = model.generator(embedding)
#             loss = criterion(log_cond_probs, inputs)
#             # assert loss is not Nan
#             assert not torch.isnan(loss), "Loss is NaN"

#             loss.backward()
#             optimizer.step()

#             print(loss.item())
#             # Stop the training loop if the loss is less than or equal to 0.1
#             if loss.item() <= 0.1:
#                 print("Loss reached the desired value of 0.1, stopping training.")
#                 break

#         else:
#             # If the inner loop wasn't broken, continue with the next epoch
#             continue

#         # If the inner loop was broken, break the outer loop as well
#         break


# # def _test_encoder_decoder_sequentially(model):
# #     c = copy.deepcopy
# #     attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
# #     ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)

# #     # define the loss function
# #     criterion = KLLoss()
# #     optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# #     src_embed_layer = nn.Linear(4, config.d_model)

# #     encoder = Encoder(
# #         EncoderLayer(config.d_model, c(attn), c(ff), config.dropout),
# #         config.num_blocks,
# #     )

# #     dec_emb_layer = nn.Linear(2, config.d_model)

# #     decoder = Decoder(
# #         DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout),
# #         config.num_blocks,
# #     )

# #     generator = Generator(config.d_model, 2)

# #     # loop over the data for the specified number of epochs
# #     for i in range(50):
# #         # for epoch in range(config.num_epochs):
# #         # set the model to train mode
# #         model.train()

# #         # loop over the training data in batches
# #         for i, batch in enumerate(train_loader):
# #             inputs, condition = batch
# #             inputs = nn.functional.one_hot(inputs, 2)
# #             inputs = inputs.to(torch.float)
# #             optimizer.zero_grad()
# #             # assert inputs.dtype == torch.int64
# #             # 1. embed the condition or prompt
# #             src_emb = src_embed_layer(condition)

# #             # 2. run the encoder
# #             encoder_out = encoder(src_emb)

# #             assert encoder_out.shape == (
# #                 config.batch_size,
# #                 config.num_atoms,
# #                 config.d_model,
# #             )

# #             # 3. embed the inputs
# #             dec_emb = dec_emb_layer(inputs)
# #             assert dec_emb.shape == (
# #                 config.batch_size,
# #                 config.num_atoms,
# #                 config.d_model,
# #             )

# #             # 4. run the decoder
# #             decoder_out = decoder(dec_emb, encoder_out)

# #             # 5. generate the output log cond probabilities
# #             log_cond_probs = generator(decoder_out)

# #             loss = criterion(log_cond_probs, inputs)
# #             print(loss)

# #             loss.backward()
# #             optimizer.step()

# #             assert not torch.isnan(loss), "Loss is NaN"
# #             # check that loss is going minized

# #             # Stop the training loop if the loss is less than or equal to 0.1
# #             if loss.item() <= 0.1:
# #                 print("Loss reached the desired value of 0.1, stopping training.")
# #                 break

# #         else:
# #             # If the inner loop wasn't broken, continue with the next epoch
# #             continue

# #         # If the inner loop was broken, break the outer loop as well
# #         break


# if __name__ == "__main__":
#     pytest.main([__file__])
#     # run pytest
#     # config, train_loader, val_loader, test_loader = init_data()
#     # model = get_rydberg_transformer(config)
#     # test_encoder_decoder(model)
#     # test_encoder_decoder_sequentially(model)
