import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import torch
from torch_geometric.nn import dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class GATwithMinCutPooling(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        num_hidden=8,
        heads=8,
        dropout=0.6,
        name="GAT",
    ):
        super(GATwithMinCutPooling, self).__init__()
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

        self.num_classes = num_classes
        self.feature_transform = torch.nn.Linear(num_classes, num_features)

    def forward(self, data, num_classes):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)

        x_dense, mask = to_dense_batch(data.x, data.batch)
        adj_dense = to_dense_adj(data.edge_index, data.batch)

        num_clusters = num_classes

        transform = torch.nn.Linear(x.size(1), num_clusters)
        s = transform(x)
        s = torch.softmax(s, dim=1)

        x_pool, adj_pool, mincut_loss, ortho_loss = dense_mincut_pool(
            x_dense, adj_dense, s, mask=mask
        )
        x_pool = x_pool.squeeze(0)

        return x, x_pool, adj_pool, mincut_loss, ortho_loss


classification_criterion = torch.nn.CrossEntropyLoss()


def train(model, data, optimizer, dataset):
    model.train()
    optimizer.zero_grad()
    x_class, x_pool, adj_pool, mincut_loss, ortho_loss = model(
        data, dataset.num_classes
    )
    classification_loss = classification_criterion(
        x_class[data.train_mask], data.y[data.train_mask]
    )
    loss = classification_loss + mincut_loss + ortho_loss
    loss.backward()
    optimizer.step()
    return loss.item(), classification_loss, mincut_loss, ortho_loss


def validate(model, data, dataset):
    model.eval()
    with torch.no_grad():
        x_class, x_pool, adj_pool, mincut_loss, ortho_loss = model(
            data, dataset.num_classes
        )
        classification_loss = classification_criterion(
            x_class[data.val_mask], data.y[data.val_mask]
        )
        loss = classification_loss + mincut_loss + ortho_loss
        return loss.item(), classification_loss, mincut_loss, ortho_loss


def test(model, data, dataset):
    model.eval()
    with torch.no_grad():
        x_class, x_pool, adj_pool, mincut_loss, ortho_loss = model(
            data, dataset.num_classes
        )
        classification_loss = classification_criterion(
            x_class[data.test_mask], data.y[data.test_mask]
        )
        loss = classification_loss + mincut_loss + ortho_loss
        return loss.item(), classification_loss, mincut_loss, ortho_loss
