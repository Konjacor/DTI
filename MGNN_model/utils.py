import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import numpy as np
import networkx as nx
from sklearn.metrics import auc as auc3
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim) - (1 / dim) * torch.ones(dim, dim)
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    # print(correct)

    correct = correct.sum()
    return correct / len(labels)


def RocAndAupr(output, labels):
    predict = []

    for i in range(len(output)):
        a = output[i].detach().numpy()
        predict.append(float(a[1]))
    # print(labels)
    # print(predict)
    c = roc_auc_score(labels, predict)
    precision, recall, thresholds = precision_recall_curve(labels, predict)
    d = auc3(recall, precision)
    return c, d


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(config):
    f = np.loadtxt(config.feature_path, dtype=float)
    l = np.loadtxt(config.label_path, dtype=int)
    test = np.loadtxt(config.test_path, dtype=int)
    train = np.loadtxt(config.train_path, dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_test


def load_graph(config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)#把文本文件中的内容转换为numpy数组，其中元素的类型是int32，feature_edges 包含特征图中所有边的信息，每行表示一条边，形如 (node1, node2)。
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)#对 feature_edges 的重新包装，确保其为 NumPy 数组并保留原始形状。
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)#使用 SciPy 的 coo_matrix 构造稀疏矩阵 fadj。
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)#确保邻接矩阵是对称的。如果(i, j)存在边但(j, i)不存在，则添加(j, i)边。
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))#给原矩阵添加单位矩阵 sp.eye(fadj.shape[0])，相当于给每个节点添加自环（给每个节点添加自环的意思是，在图的邻接矩阵中，为每个节点与自身之间添加一条边。换句话说，就是在图中让每个节点“连接到自己”。这是图神经网络（GNN）中一种常见的预处理操作。在图神经网络中，节点的更新通常依赖于它的邻居节点的信息。如果没有自环，节点可能会忽略自身的特征信息，只依赖邻居节点的影响。通过添加自环，可以让节点在更新时也考虑自己的特征）。调用 normalize 函数对邻接矩阵进行归一化处理

    #类似于特征图的加载过程，从 config.structgraph_path 文件中读取拓扑图的边信息。
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)#使用 coo_matrix 构造稀疏矩阵 sadj，表示拓扑图的邻接关系。
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)#确保拓扑图的邻接矩阵是对称的。
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))#添加自环并归一化邻接矩阵。

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)#将归一化后的拓扑图邻接矩阵转换为 PyTorch 稀疏张量格式。
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)#将归一化后的特征图邻接矩阵转换为 PyTorch 的稀疏张量格式，以便用于深度学习模型。

    return nsadj, nfadj


def get_adj(adj):
    rows = []
    cols = []
    for i in range(len(adj)):
        col = adj[i].coalesce().indices().numpy()[0]
        for j in range(len(col)):
            rows.append(i)
            cols.append(col[j])
    print("*" * 50)
    edge_index = torch.Tensor([rows, cols]).long()
    return edge_index
