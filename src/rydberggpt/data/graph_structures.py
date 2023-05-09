import networkx as nx
import numpy as np


def generate_grid_graph(n_rows, n_cols, V_0=1.0):
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
                interaction_strength = V_0 / (np.linalg.norm(pos1 - pos2) ** 6)
                graph.add_edge(node1, node2, weight=interaction_strength)

    return graph
