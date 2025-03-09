# augmentation.py

import torch
import numpy as np

class NodeDropping:
    """节点删除增广"""
    def __init__(self, p=0.1):
        self.p = p # 删除节点的概率

    def __call__(self, edge_index):
        num_nodes = torch.max(edge_index) + 1# 计算图中节点的数量
        mask = torch.rand(num_nodes) > self.p# 对每个节点生成一个随机数，并判断是否大于p
        return edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]# 保留两个端点均未被删除的边

class EdgePerturbation:
    """边扰动增广"""
    def __init__(self, p=0.1):
        self.p = p# 扰动边的概率

    def __call__(self, edge_index):
        num_edges = edge_index.shape[1]# 计算边的数量
        mask = torch.rand(num_edges) > self.p# 对每条边生成一个随机数，并判断是否大于p
        return edge_index[:, mask]# 保留随机数大于p的边

class AugmentationPipeline:
    """增广管道"""
    def __init__(self, node_drop_prob=0.1, edge_perturb_prob=0.1):
        self.node_dropper = NodeDropping(p=node_drop_prob)# 初始化节点删除增强器
        self.edge_perturber = EdgePerturbation(p=edge_perturb_prob)# 初始化边扰动增强器

    def __call__(self, edge_index):
        # 应用节点删除和边扰动
        edge_index = self.node_dropper(edge_index)
        edge_index = self.edge_perturber(edge_index)
        return edge_index