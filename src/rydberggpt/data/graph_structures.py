import networkx as nx
import numpy as np

from rydberggpt.data.dataclasses import BaseGraph


def get_graph(config: BaseGraph) -> nx.Graph:
    """
    Generates a graph based on the given configuration.

    Args:
        config (BaseGraph): The graph configuration, an instance of a subclass of the BaseGraph dataclass.

    Returns:
        (nx.Graph): The generated graph based on the configuration.

    Raises:
        NotImplementedError: If the graph name provided in the configuration is not implemented.
    """
    if config.graph_name == "grid_graph":
        graph = generate_grid_graph(config.n_rows, config.n_cols)

    else:
        raise NotImplementedError(f"Graph name {config.graph_name} not implemented.")

    return graph


def generate_grid_graph(n_rows: int, n_cols: int) -> nx.Graph:
    """
    Generates a fully connected grid graph with weights based on the reciprocal of Euclidean distance. Coordinates is in units of lattice constant a.

    Args:
        n_rows (int): The number of rows in the grid.
        n_cols (int): The number of columns in the grid.

    Returns:
        (nx.Graph): The generated grid graph with node positions and edge weights.
    """

    # Create an empty graph
    graph = nx.Graph()

    # Add nodes with positions as attributes
    for i in range(n_rows):
        for j in range(n_cols):
            node_id = i * n_cols + j
            graph.add_node(node_id, pos=(i, j))

    # Add fully connected edges with weights as the reciprocal of Euclidean distance
    for node1 in graph.nodes:
        pos1 = np.array(graph.nodes[node1]["pos"])
        for node2 in graph.nodes:
            if node1 != node2:
                pos2 = np.array(graph.nodes[node2]["pos"])
                interaction_strength = np.linalg.norm(pos1 - pos2) ** (-6)
                graph.add_edge(node1, node2, weight=interaction_strength)

    return graph
