import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, Entities, Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
import site

site.addsitedir("../")

from training.utils import *
from training.graphmlp_utils import *
from models import gcn, graphsage, graphmlp, graphsaint, mlp, gat


dataset = load_dataset("Cora")

model_classes = [
    # gcn.GCN,
    # graphsage.SAGE,
    # graphsaint.SAINT,
    gat.GAT,
    # graphmlp.GMLP,
    # mlp.MLP,
]

model_dict = {}
optimizer_dict = {}

for model_class in model_classes:
    model_name = model_class.__name__.lower()
    model = model_class(
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        num_hidden=32,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model_dict[f"{model_name}_model"] = model
    optimizer_dict[f"{model_name}_optimizer"] = optimizer

    if model_name == "gmlp":
        gmlp_loop(model, optimizer, dataset, silent=True)
    else:
        loop(model, optimizer, dataset, silent=False)
