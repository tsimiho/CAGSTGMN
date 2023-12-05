import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch_geometric.datasets import TUDataset, Entities, Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import f1_score, normalized_mutual_info_score
import numpy as np
import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from time import perf_counter


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def load_dataset(name, root="../data"):
    if name.lower() == "aifb":
        dataset = Entities(root=root, name="AIFB", transform=NormalizeFeatures())
    elif name.lower() == "mutag":
        dataset = TUDataset(root=root, name="MUTAG", transform=NormalizeFeatures())
    elif name.lower() == "cora":
        dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
    else:
        raise Exception("Invalid Dataset!")

    return dataset


def train(model, optimizer, data, criterion=torch.nn.CrossEntropyLoss()):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test(model, data, mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    f1 = f1_score(data.y[mask], pred[mask], average="weighted")
    nmi = normalized_mutual_info_score(data.y[mask], pred[mask])
    return acc, f1, nmi


def loop(model, optimizer, dataset, silent=False):
    for data in dataset:
        for epoch in range(1, 201):
            loss = train(model, optimizer, data)
            val_acc, val_f1, val_nmi = test(model, data, data.val_mask)
            test_acc, test_f1, test_nmi = test(model, data, data.test_mask)
            if epoch % 10 == 0 and not silent:
                print(
                    f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_Acc: {val_acc:.4f}, Test_Acc: {test_acc:.4f}, Val_F1: {val_f1:.4f}, Test_F1: {test_f1:.4f}, Val_NMI: {val_nmi:.4f}, Test_NMI: {test_nmi:.4f}"
                )

        acc, f1, nmi = test(model, data, data.test_mask)
        print(
            f"{model.name} - Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, NMI: {nmi:.4f}"
        )
