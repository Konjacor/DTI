# contrastive_learning.py

import torch
import torch.nn.functional as F

class ContrastiveLearning:
    """对比学习模块"""
    def __init__(self, temperature=0.1, contrastive_weight=0.1):
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

    def compute_contrastive_loss(self, view1_embeddings, view2_embeddings):
        """计算对比损失"""
        z1 = F.normalize(view1_embeddings, dim=-1)
        z2 = F.normalize(view2_embeddings, dim=-1)

        similarity_matrix = torch.mm(z1, z2.t()) / self.temperature
        batch_size = z1.size(0)

        labels = torch.arange(batch_size).to(z1.device)
        loss = (F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.t(), labels)) / 2

        return loss * self.contrastive_weight