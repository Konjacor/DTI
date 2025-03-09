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
    # print(f.shape)#(2664, 500)(2664, 500)(2664, 500)(2664, 500)(2664, 500)
    l = np.loadtxt(config.label_path, dtype=int)
    # print(l.shape)#(2664,)(2664,)(2664,)(2664,)(2664,)
    test = np.loadtxt(config.test_path, dtype=int)
    # print(test.shape)#(222,)(222,)(222,)(222,)(222,)
    train = np.loadtxt(config.train_path, dtype=int)
    # print(train.shape)#(2000,)(2000,)(2000,)(2000,)(2000,)
    features = sp.csr_matrix(f, dtype=np.float32)#将特征数据转换为 SciPy 的压缩稀疏行矩阵（CSR 格式），适用于稀疏数据处理。
    # print(features.shape)#(2664, 500)(2664, 500)(2664, 500)(2664, 500)(2664, 500)
    features = torch.FloatTensor(np.array(features.todense()))#将 CSR 矩阵转换回密集矩阵（NumPy 数组）后将 NumPy 数组转换为 PyTorch 的浮点张量（FloatTensor）。
    # print(features.shape)#torch.Size([2664, 500])torch.Size([2664, 500])torch.Size([2664, 500])torch.Size([2664, 500])torch.Size([2664, 500])

    idx_test = test.tolist()#将 NumPy 数组转换为 Python 列表。
    # print(len(idx_test))#222 222 222 222 222
    idx_train = train.tolist()#将 NumPy 数组转换为 Python 列表。
    # print(len(idx_train))#2000 2000 2000 2000 2000

    idx_train = torch.LongTensor(idx_train)#将 Python 列表转换为 PyTorch 的长整型张量（LongTensor）
    # print(idx_train.shape)#torch.Size([2000])torch.Size([2000])torch.Size([2000]) torch.Size([2000])torch.Size([2000])
    idx_test = torch.LongTensor(idx_test)#将 Python 列表转换为 PyTorch 的长整型张量（LongTensor）
    # print(idx_test.shape)#torch.Size([222])torch.Size([222])torch.Size([222])torch.Size([222])torch.Size([222])

    label = torch.LongTensor(np.array(l))#将 NumPy 数组转换为 PyTorch 的长整型张量。
    # print(label.shape)#torch.Size([2664])torch.Size([2664])torch.Size([2664])torch.Size([2664])torch.Size([2664])

    return features, label, idx_train, idx_test


def load_graph(config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'

    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)#把文本文件中的内容转换为numpy数组，其中元素的类型是int32，feature_edges 包含特征图中所有边的信息，每行表示一条边，形如 (node1, node2)。
    # print(feature_edges.shape)#(2271, 2)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)#对 feature_edges 的重新包装，确保其为 NumPy 数组并保留原始形状。
    # print(fedges.shape)#(2271, 2)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)#使用 SciPy 的 coo_matrix 构造稀疏矩阵 fadj。
    # print(fadj.shape)#(2664, 2664)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)#确保邻接矩阵是对称的。如果(i, j)存在边但(j, i)不存在，则添加(j, i)边。
    # print(fadj.shape)#(2664, 2664)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))#给原矩阵添加单位矩阵 sp.eye(fadj.shape[0])，相当于给每个节点添加自环（给每个节点添加自环的意思是，在图的邻接矩阵中，为每个节点与自身之间添加一条边。换句话说，就是在图中让每个节点“连接到自己”。这是图神经网络（GNN）中一种常见的预处理操作。在图神经网络中，节点的更新通常依赖于它的邻居节点的信息。如果没有自环，节点可能会忽略自身的特征信息，只依赖邻居节点的影响。通过添加自环，可以让节点在更新时也考虑自己的特征）。调用 normalize 函数对邻接矩阵进行归一化处理
    # print(nfadj.shape)#(2664, 2664)

    #类似于特征图的加载过程，从 config.structgraph_path 文件中读取拓扑图的边信息。
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    # print(struct_edges.shape)#(10338, 2)(10794, 2)(10416, 2)(10787, 2)(10479, 2)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    # print(sedges.shape)#(10338, 2)(10794, 2)(10416, 2)(10787, 2)(10479, 2)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)#使用 coo_matrix 构造稀疏矩阵 sadj，表示拓扑图的邻接关系。
    # print(sadj.shape)#(2664, 2664)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)#确保拓扑图的邻接矩阵是对称的。
    # print(sadj.shape)#(2664, 2664)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))#添加自环并归一化邻接矩阵。
    # print(nsadj.shape)#(2664, 2664)

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)#将归一化后的拓扑图邻接矩阵转换为 PyTorch 稀疏张量格式。
    # print(nsadj.shape)#torch.Size([2664, 2664])
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)#将归一化后的特征图邻接矩阵转换为 PyTorch 的稀疏张量格式，以便用于深度学习模型。
    # print(nfadj.shape)#torch.Size([2664, 2664])

    return nsadj, nfadj


def get_adj(adj):
    #rows 和 cols 列表用于存储图中每条边的起点和终点节点索引。
    rows = []
    cols = []

    #把所有非0元素的行列坐标分别存到rows和cols中
    for i in range(len(adj)):#遍历邻接矩阵的每一行
        # 将当前行转换为 coalesced 形式并获取其非零元素的列索引
        col = adj[i].coalesce().indices().numpy()[0]
        # 对于每个非零元素的列索引，记录对应的行和列索引
        for j in range(len(col)):
            rows.append(i)
            cols.append(col[j])
    print("*" * 50)# 打印分隔符
    # 将行和列索引列表转换为 PyTorch 张量
    edge_index = torch.Tensor([rows, cols]).long()
    # print(edge_index.shape)
    return edge_index
