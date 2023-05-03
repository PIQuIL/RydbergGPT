import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# TODO write similar to get_rydberg_encoder_decoder
class GraphEmbedding(torch.nn.Module):
    def __init__(self, in_node_dim, d_hidden, d_model, num_layers):
        super(GraphEmbedding, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_node_dim, d_hidden))

        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(d_hidden, d_hidden))

        self.layers.append(GCNConv(d_hidden, d_model))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index, edge_attr))

        x = self.layers[-1](x, edge_index, edge_attr)

        # Reshape the tensor here
        x = x.view(data.num_graphs, -1, x.size(-1))

        # print(type(x))
        return x
