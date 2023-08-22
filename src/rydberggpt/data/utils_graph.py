import json
from typing import Dict, List, Optional, Union

import networkx as nx
import torch
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data


def gen_pyg_data(
    graph: nx.Graph, node_features: torch.Tensor, global_features: torch.Tensor
) -> Data:
    """
    Generates PyTorch Geometric Data object from edge features (defined by NetworkX graph), node features and global features.

    Args:
        graph: NetworkX graph object with the edge features of the graph.
        node_features: Features of the nodes of the graph
        global_features: Global features of the graph.

    Returns:
        A PyTorch Geometric Data object representing the input graph.
    """

    N = len(graph_nodes())

    assert x_node.shape[0] == N

    x_node = node_features
    x_global = global_features.repeat(N, 1)  # Map global feat. to node feat.
    x = torch.cat([x_node, x_global], dim=(-1))  # Combine global and node feat.

    # Convert the edge list to a PyTorch Geometric edge_index tensor
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Get edge weights from the graph
    edge_weight = torch.tensor(
        list(nx.get_edge_attributes(graph, "weight").values()), dtype=torch.float
    )

    # Create a Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    return data


def networkx_to_pyg_data(graph: nx.Graph, node_features: torch.Tensor) -> Data:
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.

    Args:
        graph: NetworkX graph object.

    Returns:
        A PyTorch Geometric Data object representing the input graph.
    """

    x = node_features.repeat(len(graph.nodes()), 1)

    # Convert the edge list to a PyTorch Geometric edge_index tensor
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Get edge weights from the graph
    edge_weight = torch.tensor(
        list(nx.get_edge_attributes(graph, "weight").values()), dtype=torch.float
    )

    # Create a Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    return data


def batch_pyg_data(data_list: List[Data]) -> Data:
    """
    Batch a list of PyTorch Geometric Data objects into a single Data object.

    Args:
        data_list: List of PyTorch Geometric Data objects.

    Returns:
        A single batched Data object containing all input Data objects.
    """
    batched_data = PyGBatch.from_data_list(data_list)
    return batched_data


def graph_to_dict(graph: nx.Graph) -> Dict:
    """
    Convert a NetworkX graph to a dictionary.

    Args:
        graph: NetworkX graph object.

    Returns:
        A dictionary representing the NetworkX graph.
    """
    graph_dict = nx.node_link_data(graph)
    return graph_dict


def save_graph_to_json(graph_dict: Dict, file_path: str) -> None:
    """
    Save a dictionary representing a NetworkX graph to a JSON file.

    Args:
        graph_dict: Dictionary representing a NetworkX graph.
        file_path: Path to the JSON file to save.
    """
    with open(file_path, "w") as f:
        json.dump(graph_dict, f)


def read_graph_from_json(file_path: str) -> Dict:
    """
    Read a JSON file and convert it to a dictionary representing a NetworkX graph.

    Args:
        file_path: Path to the JSON file to read.

    Returns:
        A dictionary representing a NetworkX graph.
    """
    with open(file_path, "r") as f:
        graph_dict = json.load(f)
    return graph_dict


def dict_to_graph(graph_dict: Dict) -> nx.Graph:
    """
    Create a NetworkX graph from a dictionary.

    Args:
        graph_dict: Dictionary representing a NetworkX graph.

    Returns:
        NetworkX graph object.
    """
    graph = nx.node_link_graph(graph_dict)
    return graph
