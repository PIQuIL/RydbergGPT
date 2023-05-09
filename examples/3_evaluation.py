import networkx as nx
import torch
from torch_geometric.data import Batch

from rydberggpt.data.dataclasses import BaseGraph, GridGraph
from rydberggpt.data.graph_structures import generate_grid_graph
from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.data.loading.utils_graph import networkx_to_pyg_data
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.training.utils import get_ckpt_path
from rydberggpt.utils import create_config_from_yaml, load_yaml_file
from rydberggpt.utils_ckpt import get_model_from_ckpt

from_ckpt = 25

log_path = get_ckpt_path(from_ckpt=from_ckpt)
# config
yaml_dict = load_yaml_file(log_path, "hparams.yaml")
config = create_config_from_yaml(yaml_dict)
# model

model = get_model_from_ckpt(
    log_path,
    model=get_rydberg_graph_encoder_decoder(config),
)


num_samples = 10
num_atoms = 20

train_loader, val_loader = get_rydberg_dataloader(
    config.batch_size, test_size=0.2, num_workers=config.num_workers
)
# get first batch of data


def generate_graph(
    config: BaseGraph,
) -> nx.Graph:
    if config.graph_name == "grid_graph":
        graph = generate_grid_graph(config.n_rows, config.n_cols, config.V_0)
    else:
        raise NotImplementedError(f"Graph name {config.graph_name} not implemented.")
    return graph


graph_config = GridGraph(
    num_atoms=num_atoms,
    graph_name="grid_graph",
    V_0=1.0,
    delta=1.0,
    omega=1.0,
    beta=1.0,
    n_rows=4,
    n_cols=5,
)


graph = generate_graph(graph_config)

node_features = torch.tensor(
    [graph_config.delta, graph_config.omega, graph_config.beta, graph_config.V_0],
    dtype=torch.float32,
)
# print(config.in_node_dim)
assert config.in_node_dim == len(
    node_features
), "Node features do not match with input shape of the gnn"


pyg_graph = networkx_to_pyg_data(graph, node_features)
repeated_data_list = [pyg_graph.clone() for _ in range(num_samples)]
batch_graph = Batch.from_data_list(repeated_data_list)

samples = model.get_samples(
    batch_size=num_samples, cond=batch_graph, num_atoms=num_atoms
)
print(samples)
