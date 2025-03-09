import torch.nn as nn

from MGNN_model.models import SFGCN

import torch.nn.functional as F

class ResidualLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return x + self.fc(x)

# 在 SFGCN 中集成残差层
class EnhancedSFGCN(SFGCN):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(EnhancedSFGCN, self).__init__(nfeat, nclass, nhid1, nhid2, n, dropout)
        self.residual = ResidualLayer(nhid1, nhid1)#这个地方参数不对会有问题

    def forward(self, features, sadj, fadj, asadj, afadj):
        x = super().forward(features, sadj, fadj, asadj, afadj)  # 调用原始前向传播
        #x的形状：1.[2664,2]
        x = self.residual(x)  # 添加残差连接
        return F.log_softmax(x, dim=-1)#输出层（？）