# augmentation.py

import torch
import numpy as np

class NodeDropping:
    """节点删除增广"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, edge_index):
        num_nodes = torch.max(edge_index) + 1
        mask = torch.rand(num_nodes) > self.p
        return edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]

class EdgePerturbation:
    """边扰动增广"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, edge_index):
        num_edges = edge_index.shape[1]
        mask = torch.rand(num_edges) > self.p
        return edge_index[:, mask]

class AugmentationPipeline:
    """增广管道"""
    def __init__(self, node_drop_prob=0.1, edge_perturb_prob=0.1):
        self.node_dropper = NodeDropping(p=node_drop_prob)
        self.edge_perturber = EdgePerturbation(p=edge_perturb_prob)

    def __call__(self, edge_index):
        # 应用节点删除和边扰动
        edge_index = self.node_dropper(edge_index)
        edge_index = self.edge_perturber(edge_index)
        return edge_index