from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from gmn_utils import *
from configure import *
import numpy as np
import torch.nn as nn
import collections
import time
import os
from datetime import datetime

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Print configure
config = get_default_config()
for k, v in config.items():
    print("%s= %s" % (k, v))

# Set random seeds
seed = config["seed"]
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


training_set, validation_set = build_datasets(config)

if config["training"]["mode"] == "pair":
    training_data_iter = training_set.pairs(config["training"]["batch_size"])
    first_batch_graphs, _ = next(training_data_iter)
else:
    training_data_iter = training_set.triplets(config["training"]["batch_size"])
    first_batch_graphs = next(training_data_iter)

node_feature_dim = first_batch_graphs.node_features.shape[-1]
edge_feature_dim = first_batch_graphs.edge_features.shape[-1]

model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

accumulated_metrics = collections.defaultdict(list)

training_n_graphs_in_batch = config["training"]["batch_size"]
if config["training"]["mode"] == "pair":
    training_n_graphs_in_batch *= 2
elif config["training"]["mode"] == "triplet":
    training_n_graphs_in_batch *= 4
else:
    raise ValueError("Unknown training mode: %s" % config["training"]["mode"])

t_start = time.time()

model.load_state_dict(torch.load("model.pth"))

model.eval()

