import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, MinCutPoolingLoss, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class GATWithMinCutPooling(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATWithMinCutPooling, self).__init__()
        # Define GAT layers
        self.gat1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.gat2 = GATConv(8 * 8, out_channels, heads=1, dropout=0.6)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GAT layers
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)

        # Get assignment tensor for MinCut pooling
        s = torch.softmax(x, dim=1)

        # Convert to dense representation
        adj = to_dense_adj(edge_index, batch=batch)
        x, mask = to_dense_batch(x, batch=batch)

        # Apply MinCut pooling
        x_pool, adj_pool, _, _ = dense_mincut_pool(x, adj, s, mask=mask, temp=1.0)

        return x_pool, adj_pool, s


# Load Cora dataset
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

# Initialize the model, optimizer, and loss functions
model = GATWithMinCutPooling(
    in_channels=dataset.num_node_features, out_channels=dataset.num_classes
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
classification_criterion = torch.nn.CrossEntropyLoss()
mincut_criterion = MinCutPoolingLoss()


# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    x_pool, adj_pool, s = model(data)
    classification_loss = classification_criterion(x_pool, data.y)
    mincut_loss = mincut_criterion(x_pool, adj_pool, s, data.batch)
    loss = classification_loss + mincut_loss
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(200):
    loss = train()
    print(f"Epoch {epoch}: Loss {loss}")
