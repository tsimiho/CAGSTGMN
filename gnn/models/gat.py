"""
gat.py: GAT model based on torch_geometric.nn.GATConv layers
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_hidden=8,
        heads=8,
        dropout=0.6,
        name="GAT",
    ):
        super(GAT, self).__init__()
        self.name = name

        self.conv1 = GATConv(
            in_channels=num_features,
            out_channels=num_hidden,
            heads=heads,
            dropout=dropout,
        )

        self.conv2 = GATConv(
            in_channels=num_hidden * heads,
            out_channels=num_classes,
            heads=1,
            dropout=dropout,
        )

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x
