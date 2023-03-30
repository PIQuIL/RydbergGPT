import os
import sys

sys.path.append(os.path.abspath(".."))

########################################################################################

import rydberggpt as rgpt

import torch

########################################################################################


N = 10
D = 1000

x = torch.rand((D, N, 2), dtype=torch.float)

########################################################################################

N_emb, N_head, N_block = 2, 1, 1


model = rgpt.TransformerEncoder(N_emb, N_head, N_block)

model = rgpt.TransformerDecoder(N_emb, N_head, N_block)

print(model([x,x]))
