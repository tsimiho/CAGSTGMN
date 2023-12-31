import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch_geometric.datasets import TUDataset, Entities, Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from time import perf_counter


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


def compute_mincut_loss(adj, s, cluster_balance_param=1.0):
    """
    Compute the MinCut pooling loss.

    :param x: Node features (after pooling).
    :param adj: Adjacency matrix of the graph (after pooling).
    :param s: Soft cluster assignment matrix.
    :param cluster_balance_param: Hyperparameter to balance cluster sizes.
    :return: MinCut pooling loss.
    """

    adj_pooled = torch.matmul(torch.matmul(s.transpose(0, 1), adj), s)

    d_pooled = torch.diag_embed(adj_pooled.sum(dim=1))

    mincut_numerator = torch.trace(
        torch.matmul(torch.matmul(s.transpose(0, 1), d_pooled - adj), s)
    )
    mincut_denominator = torch.trace(
        torch.matmul(torch.matmul(s.transpose(0, 1), d_pooled), s)
    )
    mincut_loss = mincut_numerator / mincut_denominator

    ss_t = torch.matmul(s, s.transpose(0, 1))
    i = torch.eye(ss_t.size(0), dtype=ss_t.dtype, device=ss_t.device)
    ortho_loss = torch.norm(ss_t - i, p="fro")

    total_loss = mincut_loss + cluster_balance_param * ortho_loss

    return total_loss


def train(model, optimizer, data, criterion=torch.nn.CrossEntropyLoss(), alpha=0.5):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    x, adj, s, mc_loss = model(data)

    # Compute node classification loss
    node_class_loss = criterion(x[data.train_mask], data.y[data.train_mask])

    # Compute MinCut pooling loss
    mincut_loss = compute_mincut_loss(adj, s, cluster_balance_param=1.0)

    # Combine losses
    total_loss = (1 - alpha) * node_class_loss + alpha * mincut_loss

    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()

    return total_loss


def loop(model, optimizer, dataset, epochs=200, alpha=0.5, silent=False):
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data in dataset:
            loss = train(model, optimizer, data, alpha=alpha)
            total_loss += loss.item()

        average_loss = total_loss / len(dataset)
        if not silent and epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {average_loss:.4f}")


def test(model, data, mask):
    model.eval()

    x, adj, s = model(data)

    pred = x.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    f1 = f1_score(data.y[mask], pred[mask], average="weighted")

    return acc, f1


def loop(model, optimizer, dataset, epochs=200, alpha=0.5, silent=False):
    for epoch in range(1, epochs + 1):
        total_loss = 0
        model.train()

        for data in dataset:
            # Train on each graph in the dataset
            loss = train(model, optimizer, data, alpha=alpha)
            total_loss += loss.item()

        average_loss = total_loss / len(dataset)

        if not silent and epoch % 10 == 0:
            # Print the average loss every 10 epochs
            print(f"Epoch {epoch}/{epochs}, Loss: {average_loss:.4f}")

        # Optional: Evaluate the model on validation and test sets
        if not silent and epoch % 10 == 0:
            validate_and_test(model, dataset)


def validate_and_test(model, dataset):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            # Assuming the model outputs are in the format: x, adj, s
            x, adj, s = model(data)

            # If your task is node classification, you might do something like this:
            pred = x.argmax(dim=1)

            # Validation metrics
            val_correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
            val_f1 = f1_score(
                data.y[data.val_mask].cpu(),
                pred[data.val_mask].cpu(),
                average="weighted",
            )

            # Test metrics
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
            test_f1 = f1_score(
                data.y[data.test_mask].cpu(),
                pred[data.test_mask].cpu(),
                average="weighted",
            )

            print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
