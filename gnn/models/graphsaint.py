"""
graphsaint.py: based on GraphSAINT model which is based on torch_geometric.nn.GraphConv layers
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class SAINT(torch.nn.Module):
    def __init__(
        self,
        num_features,  # num_node_features
        num_classes,
        num_hidden=32,
        dropout=0.5,
        name="GraphSAINT",
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
            self.conv1 = GraphConv(num_features, num_classes)
        else:
            self.conv1 = GraphConv(num_features, num_hidden)
            self.conv2 = GraphConv(num_hidden, num_hidden)
            self.lin = torch.nn.Linear(2 * num_hidden, num_classes)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr

    def forward(self, data, edge_weight=None):
        if self.num_hidden == 0:
            x = self.conv1(data.x, data.edge_index, edge_weight)
        else:
            x1 = F.relu(self.conv1(data.x, data.edge_index, edge_weight))
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.relu(self.conv2(x1, data.edge_index, edge_weight))
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            x = torch.cat([x1, x2], dim=-1)
            x = self.lin(x)
        return F.log_softmax(x, dim=-1)
