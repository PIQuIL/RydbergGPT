import os
import sys

sys.path.append(os.path.abspath(".."))


########################################################################################

import torch
import torch.optim as optim

import rydberggpt as rgpt

########################################################################################

# add random seed
torch.manual_seed(0)


num_atoms = 3  # number of atoms
num_samples = 8  # number of samples

H = torch.rand(
    (num_samples, num_atoms, 4), dtype=torch.float
)  # [batch_size , num_atoms, 4]
dataset = torch.randint(0, 2, (num_samples, num_atoms), dtype=torch.int64)

# print(H.shape, H)
# print(dataset.shape, dataset)

########################################################################################

N_emb, N_head, N_block = 10, 2, 2

model = rgpt.TransformerWavefunction(N_emb, N_head, N_block)

y = model([H, dataset])
# P = model.probs([H, dataset])
# print(P.shape, P)

# define loss function

from rydberggpt.models.transformer.loss import LabelSmoothing

# loss = loss_fn(P)
# print(loss)
# loss = model.loss_fn([H, dataset])
# print(loss)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = LabelSmoothing(0.1)

# optimize model
# H = torch.rand((num_samples, num_atoms, 2), dtype=torch.float)
# print(H.shape, "shape")


counter = 0
while True:
    # print(dataset.shape)  # [batch_size , num_atoms]
    # print(H.shape)  # [batch_size , num_atoms, 4]
    data = dataset
    # print(data.shape)
    optimizer.zero_grad()
    # prev_cond_p = H # H is a tensor with grads. If I feed it in again I have recursive gradients.
    # H = model([H, data])
    # assert H.shape == prev_cond_p.shape, "H shape changed"
    # print(H.shape, "condpshape")
    # print()
    # loss = model.dataloss([H, data])
    # print(loss)

    # log_probs = model.get_log_probs([H, data])
    cond_probs = model.forward([H, data])
    loss = criterion(cond_probs, data)

    # P = model.P([H, data])
    # loss = loss_fn(P)
    loss.backward()
    optimizer.step()
    print(loss)

    counter += 1
    if counter > 100:
        break
