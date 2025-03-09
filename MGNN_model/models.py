import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GAL
import torch


class GCN(nn.Module):#图卷积网络
    #500 256 64 0.2
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)#第一层图卷积
        self.gc2= GraphConvolution(nhid, nhid2)#第二层图卷积

        self.dropout = dropout

    def forward(self, x, adj):
        #经过第一层图卷积后应用ReLU激活函数，然后进行dropout。
        # print(x.shape)#1.[2664, 500]
        # print(adj.shape)#1.[2664, 2664]
        x = F.relu(self.gc1(x, adj))
        # print(x.shape)#1.[2664, 256]
        x = F.dropout(x, self.dropout, training=self.training)
        # print(x.shape)#1.[2664, 256]
        #经过第二层图卷积后再次应用ReLU激活函数和dropout。
        x = F.relu(self.gc2(x, adj))
        # print(x.shape)#1.[2664, 64]
        x = F.dropout(x, self.dropout, training=self.training)
        # print(x.shape)#1.[2664, 64]
        return x


class Attention(nn.Module):#注意力机制，用于加权融合不同来源的节点嵌入
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        #in_size的大小：1. 64
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),# 线性变换：将输入特征数量从 in_size 转换为 hidden_size
            nn.Tanh(),# 使用 Tanh 激活函数
            nn.Linear(hidden_size, 1, bias=False)# 再次线性变换：将 hidden_size 转换为 1 维输出（即每个源的重要性权重）
        )#多层感知机（MLP），用于将输入转换为权重向量

    def forward(self, z):
        #z的形状：1.[2664, 3, 64]
        #使用MLP计算每个源的重要性权重 beta，然后对所有源进行加权求和得到最终嵌入。
        #w的形状：1.[2664, 3, 1]
        w = self.project(z)
        #beta的形状：1.[2664, 3, 1]，表示每组三个嵌入的权重
        beta = torch.softmax(w, dim=1)
        #将 beta 和原始嵌入 z 相乘(逐元素相乘，广播机制把beta的第三位扩展到64的大小了)，得到每个嵌入的加权版本。相乘的结果仍然是一个形状为 [2664, 3, 64] 的张量。
        #使用 .sum(1) 对第二个维度（即 dim=1）求和，将每个节点的三个加权嵌入合并为一个单一的嵌入向量，形状变为 [2664, 64]
        return (beta * z).sum(1), beta


class SFGCN(nn.Module):#结合了GCN和GAT的优点，并通过注意力机制融合信息。
    #500 2 256 64 2664 0.2
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(SFGCN, self).__init__()
        # print(str(nfeat)+" "+str(nclass)+" "+str(nhid1)+" "+str(nhid2)+" "+str(n)+" "+str(dropout))
        self.SGAT1 = GAT(nfeat, nhid1, nhid2, dropout)#图注意力网络1
        self.SGAT2 = GAT(nfeat, nhid1, nhid2, dropout)#图注意力网络2
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)#图神经网络
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))#用于注意力机制的参数，一开始是一个形状为(nhid2,1)的全零张量

        self.residual = ResidualLayer(nhid2, nhid2)
        self.attention2 = AttentionLayer(nhid2, nhid2 // 2)

        #对张量 self.a 使用 Xavier 均匀初始化方法进行初始化，并应用一个增益值（gain）为 1.414。
        '''
        gain 参数：用来缩放初始化范围。它通常设置为与使用的激活函数相关的值。
        在这个例子中，显式地设置了 gain=1.414。这个特定值（约等于根号2）通常是为 ReLU 及其变种（如 Leaky ReLU）准备的，
        因为这些激活函数会导致前向传播时信号的方差减少大约一半，所以需要适当放大初始权重的标准差来补偿这一点。
        '''
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)#注意力机制，用于加权融合不同来源的节点嵌入
        #nn.Tanh 是一个实现双曲正切（Hyperbolic Tangent, Tanh）激活函数的类。Tanh 函数是一种非线性激活函数，其输出范围在 (-1, 1) 之间。
        self.tanh = nn.Tanh()
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, 16),
            nn.Tanh(),
            nn.Linear(16, nclass),
            nn.LogSoftmax(dim=1)
        )#用于最终分类的MLP

    def forward(self, x, sadj, fadj, asadj, afadj):
        #分别使用不同的图结构（如结构图sadj、特征图fadj等）通过不同的模型（SGAT1, SGAT2, CGCN）提取特征。
        #emb1是拓扑图（？）经过多头图注意力网络处理之后的特征数据（好像也是聚合了邻居节点的信息？）
        emb1 = self.SGAT1(x, asadj)  # Special_GAT out1 -- sadj structure graph
        # print(emb1.shape)#1.[2664, 64]
        #com1是拓扑图邻接矩阵经过图卷积神经网络聚合节点及其邻居的信息更新节点的特征表示后的特征矩阵
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        # print(com1.shape)#1.[2664, 64]
        #com2是特征图邻接矩阵经过图卷积神经网络聚合节点及其邻居的信息更新节点的特征表示后的特征矩阵
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        # print(com2.shape)#1.[2664, 64]
        #emb1是特征图（？）经过多头图注意力网络处理之后的特征数据（好像也是聚合了邻居节点的信息？）
        emb2 = self.SGAT2(x, afadj)  # Special_GAT out2 -- fadj feature graph
        # print(emb2.shape)#1.[2664, 64]
        #将不同来源的特征进行平均并堆叠，然后通过注意力机制进行加权融合。
        Xcom = (com1 + com2) / 2
        # print(Xcom.shape)#1.[2664, 64]

        #torch.stack：这个函数用于在指定的维度上连接一系列张量。
        #与 torch.cat 不同，torch.stack 会在指定的新维度上进行堆叠，而不是简单地连接现有的维度。所有被堆叠的张量必须具有相同的形状。
        #对于每个节点（每行），都可以访问到emb1、emb2、Xcom三种特征表示
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        # print(emb.shape)#1.[2664, 3, 64]
        #
        emb, att = self.attention(emb)
        # print(emb.shape)#最终的特征嵌入张量1.[2664, 64]
        # print(att.shape)#每组三个嵌入的权重张量1.[2664, 3, 1]

        emb = self.residual(emb)  # 添加残差连接
        emb = self.attention2(emb) #添加注意力层

        #最终通过MLP进行分类输出。
        output = self.MLP(emb)
        # print(output)
        # print(output.shape)#1.[2664, 2]
        return output

class GATLay(torch.nn.Module):#实现了一个多头注意力机制的GAT层。
    #in_features:500 hid_features:256 out_features:64 n_heads:4
    def __init__(self, in_features, hid_features, out_features, n_heads):
        super(GATLay, self).__init__()
        #定义了n_heads个单头注意力机制 attentions 和一个输出层 out_att
        #self.attentions 是一个列表，包含 n_heads 个 GAL 实例，每个实例负责从输入特征 in_features 到隐藏层特征 hid_features 的转换。
        self.attentions = [GAL(in_features, hid_features) for _ in
                           range(n_heads)]#500个特征->256个特征
        #self.out_att 是一个单独的 GAL 实例，用于将多个头的输出拼接结果（维度为 hid_features * n_heads）映射到最终的输出特征维度 out_features。
        self.out_att = GAL(hid_features * n_heads, out_features)#1024个特征->64个特征

    def forward(self, x, edge_index, dropout):
        #对输入 x 应用所有单头注意力机制，并将结果按照列拼接在一起（即后面的结果横着拼到后面）。
        #其中每个图注意力网络的返回值是经过图注意力网络处理后每个节点的新特征表示，大小为1.[2664,256]
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        # print(x.shape)#1.[2664,256*n_heads=1024]
        #对拼接后的结果应用dropout和ELU激活函数，最后通过输出层进行softmax归一化。
        x = F.dropout(x, dropout, training=self.training)
        # print(x.shape)#1.[2664, 1024]
        x = F.elu(self.out_att(x, edge_index))#out_att融合4个头的图注意力网络的结果
        # print(x.shape)#1.[2664, 64]
        # new = F.softmax(x, dim=1)
        # print(new.shape)#1.[2664, 64]
        # return new
        return F.softmax(x, dim=1)#1.[2664, 64]


class GAT(torch.nn.Module):#封装了一个完整的图注意力网络GAT模型。
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GAT, self).__init__()
        #包含一个 GATLay 层和一个dropout概率。
        self.gatlay = GATLay(nfeat, nhid, nhid2, 4)
        self.dropout = dropout

    def forward(self, x, adj):
        #将邻接矩阵 adj 直接作为 edge_index 输入到 GATLay 中进行处理。GATLay 需要的是边索引而不是邻接矩阵，应确保数据结构正确。
        #需要边索引是因为图注意力网络需要融合一个节点及其邻居节点的信息来更新原特征数据
        edge_index = adj
        x = self.gatlay(x, edge_index, self.dropout)#x为经过多头图注意力网络处理之后的特征数据，形状为1.[2664, 64]
        return x

class ResidualLayer(nn.Module):#残差连接层
    def __init__(self, input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return x + self.fc(x)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # print(x.shape)#[2664,3,2]
        attn_weights = torch.softmax(self.output(torch.tanh(self.attention(x))), dim=1)
        # print((attn_weights*x).shape)#1.[2664,64]
        # context_vector = torch.sum(attn_weights * x, dim=1)
        # print(context_vector.shape)#1.[2664]
        return attn_weights*x
