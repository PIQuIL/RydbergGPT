import os
import sys

sys.path.append(os.path.abspath(".."))

########################################################################################

import rydberggpt as rgpt

import torch

########################################################################################


N = 20
D = 1000

p = torch.rand((D, N, 4), dtype=torch.float)
x = torch.randint(0, 2, (D, N), dtype=torch.int64)

########################################################################################

N_emb, N_head, N_block = 10, 2, 2

model = rgpt.TransformerWavefunction(N_emb, N_head, N_block)

y = model([p, x])
P = model.P([p, x])
