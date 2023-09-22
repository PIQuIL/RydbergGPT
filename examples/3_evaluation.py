# Imports

import os
import sys

import copy

import itertools as itr

import numpy as np

import matplotlib
from matplotlib import colors, cm, patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

import networkx as nx

import torch
from torch import nn
from torch_geometric.data import Batch

import rydberggpt as rgpt
from rydberggpt.data.dataclasses import BaseGraph, GridGraph
from rydberggpt.data.graph_structures import get_graph
from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.data.utils_graph import networkx_to_pyg_data
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.observables.rydberg_energy import get_rydberg_energy
from rydberggpt.utils import create_config_from_yaml, load_yaml_file
from rydberggpt.utils_ckpt import get_ckpt_path, get_model_from_ckpt


#######################################################################################

# Package parameters

matplotlib.rcParams["figure.figsize"] = (12, 8)
matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

base_path = os.path.abspath("..")

cp = sns.color_palette("muted", 8)
cm = sns.color_palette("winter", as_cmap=True)

########################################################################################


# is_cuda = torch.cuda.is_available()
is_cuda = False
device = torch.device("cuda" if is_cuda else "cpu")
torch.set_default_device(device)

if is_cuda:
    torch.cuda.empty_cache()

########################################################################################

from_ckpt = 1

##### LOADING FROM CKPT #####
log_path = os.path.join(base_path, get_ckpt_path(from_ckpt=from_ckpt))

# config
yaml_dict = load_yaml_file(log_path, "hparams.yaml")
config = create_config_from_yaml(yaml_dict)
# model
model = get_model_from_ckpt(
    log_path, model=get_rydberg_graph_encoder_decoder(config), ckpt="best"
)

model.to(device=device)


########################################################################################


def generate_prompt(
    n_rows, n_cols, delta=1.0, omega=1.0, beta=64.0, Rb=1.15, return_graph_config=False
):
    num_atoms = n_rows * n_cols

    graph_config = GridGraph(
        num_atoms=num_atoms,
        graph_name="grid_graph",
        Rb=Rb,
        delta=delta,
        omega=omega,
        beta=beta,
        n_rows=n_rows,
        n_cols=n_cols,
    )

    graph = get_graph(graph_config)

    node_features = torch.tensor(
        [graph_config.delta, graph_config.omega, graph_config.beta, graph_config.Rb],
        dtype=torch.float32,
    )

    assert config.in_node_dim == len(
        node_features
    ), "Node features do not match with input shape of the gnn"

    pyg_graph = networkx_to_pyg_data(graph, node_features)

    if return_graph_config:
        return pyg_graph, graph_config
    else:
        return pyg_graph


########################################################################################


def snake_flip(x):
    _x = copy.deepcopy(x)

    for i in range(_x.shape[-2]):
        if i % 2 == 1:
            if isinstance(_x, torch.Tensor):
                _x[..., i, :] = torch.flip(_x[..., i, :], dims=(-1,))
            elif isinstance(_x, np.ndarray):
                _x[..., i, :] = _x[..., i, ::-1]
            else:
                assert (
                    False
                ), "Unsupported Type: Function supports only np.ndarray and torch.Tensor"
        else:
            pass

    return _x


########################################################################################

# Default parameters

L = 6
B = 1

delta = 2.0
omega = 1.0
beta = 16.0
Rb = 1.15

########################################################################################

# Generate samples for model

rows, cols = 16, 21
prompts = []
deltas = np.linspace(-2, 5, cols)
betas = np.logspace(-2, 8, rows, base=2)
for beta, delta in itr.product(betas, deltas):
    pyg_graph = generate_prompt(
        n_rows=L, n_cols=L, delta=delta, omega=omega, beta=beta, Rb=Rb
    )

    [prompts.append(pyg_graph.clone()) for _ in range(B)]

prompts = Batch.from_data_list(prompts)

samples = model.get_samples(
    batch_size=len(prompts),
    cond=prompts,
    num_atoms=L**2,
).cpu()


samples = samples[..., -1].reshape(rows, cols, B, L, L)

########################################################################################

# Generate stitched samples for plotting

images = snake_flip(samples)
images = copy.deepcopy(images)
images = torch.moveaxis(images, (0, 1), (-4, -2))
images = images.reshape(B, rows * L, cols * L)

# ########################################################################################

# Plot stitched samples

fig = plt.figure(figsize=np.array([cols, rows]) / np.array([cols, rows]).max() * 16)
ax = fig.subplots(1, 1)

ax.imshow(images[0])

xticks = (
    np.arange(0, images[0].shape[1], np.ceil(cols / 10).astype(int) * L) - 0.5 + L / 2
)
yticks = (
    np.arange(0, images[0].shape[0], np.ceil(rows / 10).astype(int) * L) - 0.5 + L / 2
)
xticklabels = np.round(deltas[:: np.ceil(cols / 10).astype(int)], 3)
yticklabels = np.round(betas[:: np.ceil(rows / 10).astype(int)], 3)

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xticklabels, fontsize=12)
ax.set_yticklabels(yticklabels, fontsize=12)

ax.set_xticks(np.arange(-0.5, cols * L, L), minor=True)
ax.set_yticks(np.arange(-0.5, rows * L, L), minor=True)

ax.set_xlabel(r"$\Delta / \Omega$")
ax.set_ylabel(r"$\beta\Omega$")


ax.grid(which="minor", color="w", linestyle="-", linewidth=1)

########################################################################################

# Calculate staggered magnetization (order parameter for checkerboard phase)

sigmas = samples - 0.5
idcs = np.indices((L, L))
checkerboard = 2 * (idcs.sum(0) % 2) - 1

staggered_magnetization = torch.abs((sigmas * checkerboard).mean((-1, -2)))

staggered_magnetization_moments = np.stack(
    [
        staggered_magnetization.mean(-1).reshape(rows, cols),
        staggered_magnetization.std(-1).rehape(rows, cols),
    ]
)

########################################################################################

# Plot staggered magnetization across the phase transition

fig = plt.figure()
ax = fig.subplots(1, 1)

c = cm(np.linspace(0, 1, rows))
for i, beta in enumerate(betas):
    ax.plot(
        deltas,
        staggered_magnetization_moments[0, i, :],
        color=c[i],
        alpha=0.75,
        label=r"$\beta={}$".format(beta),
    )
    ax.fill_between(
        deltas,
        (staggered_magnetization_moments[0] + staggered_magnetization_moments[1])[i, :],
        (staggered_magnetization_moments[0] - staggered_magnetization_moments[1])[i, :],
        color=c[i],
        alpha=0.1,
    )

ax.set_xlabel(r"$\Delta$")
ax.set_ylabel(r"$M_s$")

ax.legend(
    loc="lower center", bbox_to_anchor=(0.5, 0.95), fancybox=True, shadow=True, ncol=4
)

########################################################################################

# Calculate energy

energy_density_moments = []
for delta in deltas:
    prompts = []
    pyg_graph = generate_prompt(
        n_rows=L, n_cols=L, delta=delta, omega=omega, beta=beta, Rb=Rb
    )
    [prompts.append(pyg_graph.clone()) for _ in range(B)]

    prompts = Batch.from_data_list(prompts)

    _samples = model.get_samples(
        batch_size=len(prompts), cond=prompts, num_atoms=L**2
    )[..., -1]

    energy = get_rydberg_energy(
        model=model,
        samples=_samples,
        cond=pyg_graph,
        device=device,
        undo_sample_path=snake_flip,
    )

    energy_density_moments.append(
        torch.cat(
            [
                energy[:, 0].mean(0, keepdim=True) / L**2,
                energy[:, 0].std(0, keepdim=True) / L**2,
            ]
        )
    )

energy_density_moments = torch.stack(energy_density_moments).cpu().detach().numpy()

########################################################################################

fig = plt.figure()
ax = fig.subplots(1, 1)

ax.plot(deltas, energy_density_moments[:, 0])
ax.fill_between(
    deltas,
    energy_density_moments[:, 0] + energy_density_moments[:, 1],
    energy_density_moments[:, 0] - energy_density_moments[:, 1],
    alpha=0.25,
)

ax.set_xlabel(r"$\Delta / \Omega$")
ax.set_ylabel(r"$E / N$")
