# Imports

import networkx as nx
import torch
from torch_geometric.data import Batch

from rydberggpt.data.dataclasses import BaseGraph, GridGraph
from rydberggpt.data.graph_structures import get_graph
from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.data.utils_graph import networkx_to_pyg_data
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.observables.rydberg_energy import get_rydberg_energy
from rydberggpt.utils import create_config_from_yaml, load_yaml_file
from rydberggpt.utils_ckpt import get_ckpt_path, get_model_from_ckpt

########################################################################################

# Set pytorch device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

########################################################################################

from_ckpt = 25
##### LOADING FROM CKPT #####
log_path = get_ckpt_path(from_ckpt=from_ckpt)
# config
yaml_dict = load_yaml_file(log_path, "hparams.yaml")
config = create_config_from_yaml(yaml_dict)
# model
model = get_model_from_ckpt(
    log_path,
    model=get_rydberg_graph_encoder_decoder(config),
)
model.to(device)

########################################################################################

##### GENERATE GRAPH #####
num_samples = 5
n_rows = 3
n_cols = 3
num_atoms = n_rows * n_cols

graph_config = GridGraph(
    num_atoms=num_atoms,
    graph_name="grid_graph",
    V_0=1.0,
    delta=1.0,
    omega=1.0,
    beta=1.0,
    n_rows=n_rows,
    n_cols=n_cols,
)

graph = get_graph(graph_config)
adj_matrix = nx.adjacency_matrix(graph)
V = adj_matrix.todense()
V = torch.tensor(V, dtype=torch.float32)


node_features = torch.tensor(
    [graph_config.delta, graph_config.omega, graph_config.beta, graph_config.V_0],
    dtype=torch.float32,
)

assert config.in_node_dim == len(
    node_features
), "Node features do not match with input shape of the gnn"

pyg_graph = networkx_to_pyg_data(graph, node_features)
repeated_data_list = [pyg_graph.clone() for _ in range(num_samples)]
batch_graph = Batch.from_data_list(repeated_data_list)

########################################################################################

##### GENERATE SAMPLES #####
samples = model.get_samples(
    batch_size=num_samples, cond=batch_graph, num_atoms=num_atoms, fmt_onehot=False
)

########################################################################################

##### EVALUATE ENERGY #####
model.get_log_probs
energy = get_rydberg_energy(
    get_log_probs=model.get_log_probs,
    V=V,
    omega=graph_config.omega,
    delta=graph_config.delta,
    samples=samples,
    cond=batch_graph,
    num_atoms=num_atoms,
    device=device,
)
# print the average energy
print(f"Energy: {energy.mean().item():.4f} +/- {energy.std().item():.4f}")
