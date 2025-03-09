# contrastive_learning.py

import torch
import torch.nn.functional as F

class ContrastiveLearning:
    """对比学习模块"""
    def __init__(self, temperature=0.1, contrastive_weight=0.1):
        self.temperature = temperature#控制相似度分数的平滑程度，通常是一个较小的正值（适当调小）
        self.contrastive_weight = contrastive_weight#对比损失的权重，用于在多任务学习中平衡对比损失与其他损失（如分类损失等）（适当调大）

    def compute_contrastive_loss(self, view1_embeddings, view2_embeddings):
        """计算对比损失"""
        #对每个样本的嵌入向量进行 L2 归一化，使得每个向量的范数为 1。这一步有助于确保不同维度的特征在计算相似度时具有相同的尺度。
        z1 = F.normalize(view1_embeddings, dim=-1)
        z2 = F.normalize(view2_embeddings, dim=-1)

        #计算两个视图嵌入向量之间的相似度矩阵。这里使用了矩阵乘法 torch.mm 来计算 z1 和 z2 的点积，得到形状为 [batch_size, batch_size] 的相似度矩阵。
        #相似度矩阵中的元素 (i, j) 表示第 i 个样本在 view1 中的嵌入与第 j 个样本在 view2 中的嵌入之间的余弦相似度（经过温度缩放）。
        similarity_matrix = torch.mm(z1, z2.t()) / self.temperature
        batch_size = z1.size(0)

        #创建一个标签张量 labels，其值为从 0 到 batch_size - 1 的整数序列。这些标签表示每个样本与其自身匹配的索引。
        #由于对比学习的目标是让正样本对（即同一个样本在不同视图下的嵌入）的相似度尽可能高，而负样本对（即不同样本的嵌入）的相似度尽可能低，因此这里的标签实际上是指定了每个样本的正样本位置。
        labels = torch.arange(batch_size).to(z1.device)

        #计算两个方向上的交叉熵损失
        #第一部分将 similarity_matrix 视为预测概率分布，计算每个样本与其自身的匹配得分（即正样本对）与其他样本的匹配得分（即负样本对）之间的交叉熵损失。
        #第二部分对转置后的相似度矩阵执行同样的操作，以确保两个视图之间的对称性。
        #最终损失是这两个损失的平均值，这样可以保证双向一致性。
        loss = (F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.t(), labels)) / 2

        #将计算出的对比损失乘以 contrastive_weight 后返回。这个权重可以在多任务学习中用于平衡对比损失与其他损失函数的影响。
        return loss * self.contrastive_weight