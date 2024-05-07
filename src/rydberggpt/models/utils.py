from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from rydberggpt.data.dataclasses import GridGraph
from rydberggpt.data.graph_structures import get_graph
from rydberggpt.data.utils_graph import networkx_to_pyg_data


def generate_prompt(
    model_config: dataclass,
    n_rows: int,
    n_cols: int,
    delta: float = 1.0,
    omega: float = 1.0,
    beta: float = 64.0,
    Rb: float = 1.15,
    return_graph_config: bool = False,
) -> Data:
    graph_config = GridGraph(
        num_atoms=n_rows * n_cols,
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

    assert model_config.in_node_dim == len(
        node_features
    ), "Node features do not match with input shape of the gnn"

    pyg_graph = networkx_to_pyg_data(graph, node_features)

    if return_graph_config:
        return pyg_graph, graph_config
    else:
        return pyg_graph


def extract_model_info(module, prefix=""):
    result = {}
    for name, submodule in module.named_children():
        if prefix:
            full_name = f"{prefix}.{name}"
        else:
            full_name = name

        result[full_name] = {
            "name": full_name,
            "class": submodule.__class__.__name__,
            # "input_shape": None,
            # "output_shape": None,
            "num_parameters": sum(p.numel() for p in submodule.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in submodule.parameters() if p.requires_grad
            ),
        }
        result.update(extract_model_info(submodule, prefix=full_name))
    return result
