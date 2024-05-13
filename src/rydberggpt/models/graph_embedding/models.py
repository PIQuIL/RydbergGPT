from typing import Callable, Type

import torch
from torch import Tensor
from torch.nn import LayerNorm, ModuleList
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from rydberggpt.models.graph_embedding.layers import GraphLayer


class GraphEmbedding(torch.nn.Module):
    def __init__(
        self,
        graph_layer: Type[Callable],
        in_node_dim: int,
        d_hidden: int,
        d_model: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        """
        GraphEmbedding class for creating a graph embedding with multiple layers.

        Args:
            graph_layer (Type[Callable]): The graph layer to be used in the embedding.
            in_node_dim (int): The input node dimension. (omega, delta, beta)
            d_hidden (int): The hidden dimension size.
            d_model (int): The output node dimension.
            num_layers (int): The number of layers in the graph embedding.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(GraphEmbedding, self).__init__()

        self.graph_layer = graph_layer
        self.layers = ModuleList()
        self.layers.append(
            GraphLayer(
                self.graph_layer(in_node_dim, d_hidden), LayerNorm(d_hidden), dropout
            )
        )

        for _ in range(num_layers - 2):
            self.layers.append(
                GraphLayer(
                    self.graph_layer(d_hidden, d_hidden), LayerNorm(d_hidden), dropout
                )
            )

        self.layers.append(self.graph_layer(d_hidden, d_model))
        self.final_norm = LayerNorm(d_model)

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass through the graph embedding layers.

        Args:
            data (Data): The input graph data.

        Returns:
            (Tensor): The output tensor with reshaped dimensions.
        """
        # [..., num_features], [2, ...] [...]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for layer in self.layers[:-1]:
            # [..., num_features]
            x = layer(x, edge_index, edge_attr)

        # [..., d_model]
        x = self.final_norm(self.layers[-1](x, edge_index, edge_attr))

        x, batch_mask = to_dense_batch(x, data.batch)

        # [B, N, d_model], where N is the number of nodes or the number of atoms
        return x, batch_mask
