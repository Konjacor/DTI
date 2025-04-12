import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


# 定义多模态融合模型
class FeatureFusion(nn.Module):
    def __init__(self, seq_dim, graph_dim, hidden_dim=256):
        super().__init__()
        # 序列特征处理
        self.seq_fc = nn.Linear(seq_dim, hidden_dim)
        # 图特征处理
        self.graph_fc = nn.Linear(graph_dim, hidden_dim)
        # 注意力融合层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, seq_feat, graph_feat):
        # 投影到相同维度
        seq_proj = self.seq_fc(seq_feat)  # [batch, hidden]
        graph_proj = self.graph_fc(graph_feat)  # [batch, hidden]
        # 计算注意力权重
        combined = torch.cat([seq_proj, graph_proj], dim=1)
        attn = self.attention(combined)  # [batch, 2]
        # 加权融合
        fused = attn[:, 0:1] * seq_proj + attn[:, 1:2] * graph_proj
        return fused


# 使用示例
# seq_features = torch.randn(32, 1024)  # 批大小32，序列特征维度1024
# graph_features = torch.randn(32, 512)  # 图特征维度512
#
# model = FeatureFusion(seq_dim=1024, graph_dim=512)
# fused_features = model(seq_features, graph_features)  # 输出形状 [32, 256]