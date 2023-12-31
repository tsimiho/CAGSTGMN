import numpy as np
import torch
import site

site.addsitedir("../")

from training.utils import *
from models import gat_mincut

dataset = load_dataset("Cora")

model = gat_mincut.GATwithMinCut(
    num_classes=dataset.num_classes,
    num_features=dataset.num_features,
    num_hidden=32,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

loop(model, optimizer, dataset, silent=True)
