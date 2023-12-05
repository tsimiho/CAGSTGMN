"""
graphsage.py: GraphSAGE model based on torch_geometric.nn.SAGEConv layers
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_hidden=64,
        dropout=0.5,
        name="GraphSAGE",
    ):
        """
        if num_hidden == 0, then a 1-layer/hop model is built,
        else a 2 layer model with num_hidden is built
        """
        super().__init__()
        self.name = name
        self.num_hidden = num_hidden
        self.dropout = dropout
        if num_hidden == 0:
            self.conv1 = SAGEConv(num_features, num_classes)
        else:
            self.conv1 = SAGEConv(num_features, num_hidden)
            self.conv2 = SAGEConv(num_hidden, num_classes)

    def forward(self, data):
        if self.num_hidden == 0:
            x = self.conv1(data.x, data.edge_index)
        else:
            x = self.conv1(data.x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
            x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=-1)
