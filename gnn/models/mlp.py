"""
mlp.py: mlp model based on torch.nn.Linear layers
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLP(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, num_hidden=32, dropout=0.5, name="MLP"
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
            self.lin1 = Linear(num_features, num_classes)
        else:
            self.lin1 = Linear(num_features, num_hidden)
            self.lin2 = Linear(num_hidden, num_classes)

    def forward(self, data):
        if self.num_hidden == 0:
            x = self.lin1(data.x)
        else:
            x = self.lin1(data.x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        return F.log_softmax(x, dim=1)
