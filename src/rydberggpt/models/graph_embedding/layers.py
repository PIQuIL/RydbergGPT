import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import Adj, OptTensor


class GraphLayer(nn.Module):
    def __init__(self, graph_layer: nn.Module, norm_layer: nn.Module, dropout: float):
        """
        A GraphLayer is a single layer in a graph neural network, consisting of
        a graph layer, normalization layer, and dropout.

        Args:
            graph_layer (nn.Module): A graph layer, e.g., GCNConv, GATConv, etc.
            norm_layer (nn.Module): A normalization layer, e.g., LayerNorm or BatchNorm.
            dropout (float): Dropout probability.
        """
        super(GraphLayer, self).__init__()
        self.graph_layer = graph_layer
        self.norm = norm_layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: Adj, edge_attr: OptTensor
    ) -> torch.Tensor:
        """
        Forward pass through the GraphLayer.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (Adj): Edge indices.
            edge_attr (OptTensor): Edge feature matrix.

        Returns:
            (torch.Tensor): The output tensor after passing through the GraphLayer.
        """
        x = self.graph_layer(x, edge_index, edge_attr)
        x = F.relu(self.norm(x))
        x = self.dropout(x)
        return x
