import torch
import numpy as np
import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import f1_score, normalized_mutual_info_score


def get_data(data):
    num_nodes = data.num_nodes
    adj = torch.zeros((num_nodes, num_nodes))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    features = data.x
    labels = data.y
    idx_train = data.train_mask.nonzero(as_tuple=True)[0]
    idx_val = data.val_mask.nonzero(as_tuple=True)[0]
    idx_test = data.test_mask.nonzero(as_tuple=True)[0]
    adj_label = get_A_r(adj, 2)  # 1 or 2
    return adj, features, labels, idx_train, idx_val, idx_test, adj_label


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def fetch_normalization(type):
    switcher = {
        "AugNormAdj": aug_normalized_adjacency,  # A' = (D + I)^(-1/2) * (A+I) * (D + I)^(-1/2)
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_A_r(adj, r):
    adj_label = adj.to_dense()
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label @ adj_label
    elif r == 3:
        adj_label = adj_label @ adj_label @ adj_label
    elif r == 4:
        adj_label = adj_label @ adj_label @ adj_label @ adj_label
    return adj_label


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)

    adj = adj_normalizer(adj)

    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str="cora", normalization="AugNormAdj"):
    """
    Load Citation Networks Datasets.
    """
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, normalization)

    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


def get_batch(batch_size, data):
    """
    get a batch of feature & adjacency matrix
    """
    adj, features, labels, idx_train, idx_val, idx_test, adj_label = get_data(data)
    rand_indx = torch.tensor(
        np.random.choice(np.arange(adj_label.shape[0]), batch_size)
    ).type(torch.long)
    rand_indx[0 : len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx, :][:, rand_indx]
    return features_batch, adj_label_batch


def train(model, optimizer, data):
    adj, features, labels, idx_train, idx_val, idx_test, adj_label = get_data(data)
    features_batch, adj_label_batch = get_batch(batch_size=2048, data=data)
    model.train()
    optimizer.zero_grad()
    output, x_dis = model(features_batch)
    loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau=2.0)
    loss_train = loss_train_class + loss_Ncontrast * 10.0
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return


def test(model, data):
    adj, features, labels, idx_train, idx_val, idx_test, adj_label = get_data(data)
    model.eval()
    out = model(features)
    pred = out.argmax(dim=1)
    acc_test = accuracy(out[idx_test], labels[idx_test])
    acc_val = accuracy(out[idx_val], labels[idx_val])
    f1 = f1_score(
        labels[idx_test].cpu().numpy(), pred[idx_test].cpu().numpy(), average="weighted"
    )
    nmi = normalized_mutual_info_score(
        labels[idx_test].cpu().numpy(), pred[idx_test].cpu().numpy()
    )
    return acc_test, f1, nmi


def gmlp_loop(model, optimizer, dataset, silent=False):
    for data in dataset:
        for epoch in range(1, 201):
            loss = train(model, optimizer, data)
            val_acc, val_f1, val_nmi = test(model, data)
            test_acc, test_f1, test_nmi = test(model, data)
            if epoch % 10 == 0 and not silent:
                print(
                    f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_Acc: {val_acc:.4f}, Test_Acc: {test_acc:.4f}, Val_F1: {val_f1:.4f}, Test_F1: {test_f1:.4f}, Val_NMI: {val_nmi:.4f}, Test_NMI: {test_nmi:.4f}"
                )

        acc, f1, nmi = test(model, data)
        print(f"{model.name} - Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
