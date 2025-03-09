import torch
import torch.nn as nn

from MGNN_model.models import SFGCN

import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # print(x.shape)#[2664,3,2]
        attn_weights = torch.softmax(self.output(torch.tanh(self.attention(x))), dim=1)
        context_vector = torch.sum(attn_weights * x, dim=1)
        return context_vector

# class AttentionLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(AttentionLayer, self).__init__()
#         # 注意力机制的第一步：从 input_dim 转换到 hidden_dim
#         self.attention = nn.Linear(input_dim, hidden_dim)
#         # 注意力机制的第二步：从 hidden_dim 转换到 1
#         self.output = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         # 将 x 的形状变为 [batch_size * num_nodes, input_dim]
#         original_shape = x.shape
#         x_flattened = x.view(-1, x.shape[-1])
#
#         attn_weights = torch.softmax(
#             self.output(torch.tanh(self.attention(x_flattened))),
#             dim=1
#         )
#
#         # 将 attn_weights 转换回原始形状
#         attn_weights = attn_weights.view(original_shape[0], original_shape[1], -1)
#
#         # 计算加权求和后的上下文向量
#         context_vector = torch.sum(attn_weights * x, dim=1)
#         return context_vector

# 在 SFGCN 中集成注意力层
class EnhancedSFGCN(SFGCN):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(EnhancedSFGCN, self).__init__(nfeat, nclass ,nhid1, nhid2, n, dropout)
        # self.attention = AttentionLayer(nhid2, nhid2 // 2)

    def forward(self, features, sadj, fadj, asadj, afadj):
        x = super().forward(features, sadj, fadj, asadj, afadj)  # 调用原始前向传播
        self.attention = AttentionLayer(x.shape[1], x.shape[1]//2)
        x = self.attention(x.unsqueeze(0)).squeeze(0)  # 添加注意力机制
        return F.log_softmax(x, dim=-1)

# class EnhancedSFGCN(SFGCN):
#     def __init__(self, nfeat, nhid1, nhid2, nclass, n, dropout):
#         super(EnhancedSFGCN, self).__init__(nfeat, nhid1, nhid2, nclass, n, dropout)
#         # 使用 nhid2 作为 hidden_dim 参数
#         self.attention = AttentionLayer(2, nhid2 // 2)
#
#     def forward(self, features, sadj, fadj, asadj, afadj):
#         x = super().forward(features, sadj, fadj, asadj, afadj)  # 调用原始前向传播
#         if len(x.shape) == 3:
#             x = self.attention(x)  # 直接传递三维张量
#         else:
#             raise ValueError("Unexpected shape of x in EnhancedSFGCN.forward")
#         return F.log_softmax(x, dim=-1)