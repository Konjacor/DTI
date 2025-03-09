import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid


class GraphConvolution(Module):
    """
    Simple GCN layer
    """
    #in_features: 输入特征的数量。out_features: 输出特征的数量。bias: 是否包含偏置项，默认为 True。
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.weight: 一个可训练的权重矩阵，形状为 (in_features, out_features)。
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#1.[500,256]
        #如果 bias=True，则为一个可训练的偏置向量，形状为 (out_features,)；否则为 None。
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        #调用 reset_parameters() 方法来初始化权重和偏置。
        self.reset_parameters()

    #作用：初始化权重和偏置的值，以确保它们在合理的范围内开始训练。
    def reset_parameters(self):
        #计算标准差 stdv，用于均匀分布的范围计算。
        stdv = 1. / math.sqrt(self.weight.size(1))
        #使用 uniform_ 方法将权重和偏置初始化为[-stdv,stdv]区间内的随机值。
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #input: 节点特征矩阵，形状为 (num_nodes, in_features)。
    #adj: 邻接矩阵（稀疏格式），表示图的结构信息，形状为 (num_nodes, num_nodes)。
    #input：特征数据，形状为1.[2664,500] adj：归一化后的拓扑图/特征图邻接矩阵，形状为1.[2664,2664]
    def forward(self, input, adj):
        #线性变换：使用 torch.mm 对输入特征进行线性变换，得到 support，其形状为 (num_nodes, out_features)。
        #这种变换允许模型学习更复杂的特征表示，从而提高对不同任务的表达能力。
        #self.weight 是一个可训练的权重矩阵，通过调整这些权重，模型可以在训练过程中优化特征表示，以更好地适应目标任务（如分类、回归等）。
        support = torch.mm(input, self.weight)#weight形状为1.[500,256]
        #support形状为1.[2664,256]
        #图卷积操作：使用 torch.spmm（稀疏矩阵乘法）将邻接矩阵 adj 和 support 相乘，得到输出特征 output，其形状为 (num_nodes, out_features)。
        #图卷积操作的核心思想是聚合节点及其邻居的信息来更新节点的特征表示。这有助于捕捉图中的局部结构信息，并利用邻居节点的特征来丰富当前节点的特征表示。
        #使用 torch.spmm（稀疏矩阵乘法）而不是普通的矩阵乘法，是因为邻接矩阵 adj 通常是稀疏的（大多数图都是稀疏的）。稀疏矩阵乘法可以显著减少计算量和内存占用，提高效率。
        #为什么这样做能聚合节点及其邻居的信息？因为稀疏矩阵的一行为当前节点和其它节点的关系，用这一行去乘特征矩阵，那么得到的结果行的所有列中都会包含当前节点邻居的信息（如果两点没关系，则邻接矩阵中的值为0，在矩阵乘法中就不会对最后的结果产生影响；以此类推两点有关系的情况），以此类推，所有行都能聚合当前节点及其邻居的信息来更新当前节点的特征表示。
        output = torch.spmm(adj, support)
        #output形状为1.[2664,256]
        #添加偏置：如果存在偏置项，则将其加到输出特征上。
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    #作用：返回一个描述对象的字符串，方便调试时打印对象信息。
    def __repr__(self):
        #格式化输出为：GraphConvolution (in_features -> out_features)。
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GAL(MessagePassing):
    #in_features:500 out_features:256
    def __init__(self, in_features, out_features):
        super(GAL, self).__init__()
        #print(in_features, out_features)
        #一个可训练的参数矩阵，用于计算注意力系数。其形状为 (2 * out_features, 1)，因为需要对每一对相连节点的特征进行拼接。
        #1.[512,1]
        self.a = torch.nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        #用于以均匀分布的方式初始化张量。该方法根据输入和输出单元的数量调整初始化的范围，使得每一层的输入和输出方差尽可能保持一致。
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        # 定义leakyrelu激活函数，用于增强模型的非线性表达能力。
        self.leakyrelu = torch.nn.LeakyReLU()
        #一个线性变换层，用于将输入特征 in_features 转换为 out_features 维度。
        #主要用于执行线性变换（即矩阵乘法和加法），其数学表达式为y = xA^T+b,x 是输入张量。A 是权重矩阵，形状为 [out_features, in_features]。b 是偏置向量(但这里没有)，形状为 [out_features]。y 是输出张量。
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):#x：1.[2664, 500] edge_index：1.[2, 23338]
        # print(x.size())#1.[2664,500]
        x = self.linear(x)#线性变换：首先通过 self.linear 对输入特征 x 进行线性变换。1.x从[2664,500]变为[2664,256]
        # print(x.size())#1.[2664,256]
        N = x.size()[0]#从张量 x 中获取其第0维度的大小，并将其赋值给变量 N
        row, col = edge_index#获取边索引：edge_index 包含所有边的起点和终点节点索引。1.
        # print(row.size(), col.size())#1.[23338][23338]
        #拼接特征：对于每条边，将其两端节点的特征拼接在一起，形成 a_input。
        a_input = torch.cat([x[row], x[col]], dim=1)
        # print(a_input.size())#1.[23338,512]

        #计算注意力系数：通过矩阵乘法和激活函数计算注意力系数e。
        #1.[23338,512]*[512,1]
        temp = torch.mm(a_input, self.a).squeeze()#squeeze方法用于移除张量中大小为1的所有维度。
        #print(temp.size())#1.[23338]
        #LeakyReLU: 为了缓解 ReLU 的“死亡神经元”（即某些神经元可能永远不会被激活）问题，LeakyReLU 在负数区域引入了一个小的斜率α，对于输入x，如果x大于等于0，则输出x本身，若x小于0，则输出αx
        e = self.leakyrelu(temp)
        # print(e.size())#1.[23338]
        #归一化：对每个节点的所有入边（？）的注意力系数进行 softmax 归一化。这样做实际上是整合了一个节点及其邻居节点的信息
        #e_all 是一个长度为 N 的零向量，用于存储每个节点所有入边的指数和
        e_all = torch.zeros(N)
        #print(e_all.size())#1.[2664]
        #对于每条边 i，我们将 e[i] 的指数值累加到 e_all 中对应起点节点的位置 row[i] 上。
        for i in range(len(row)):
            e_all[row[i]] += math.exp(e[i])

        # f = open("atten.txt", "w")

        #对于每条边 i，将 e[i] 转换为其归一化的注意力系数。
        #具体来说，对于每条边 i，其归一化的注意力系数是 exp(e[i]) / sum(exp(e[j]))，其中 j 是所有以 row[i] 为起点的边的索引集合。
        for i in range(len(e)):
            e[i] = math.exp(e[i]) / e_all[row[i]]
        #     f.write("{:.4f}\t {} \t{}\n".format(e[i], row[i], col[i]))
        #
        # f.close()
        #消息传递：调用 propagate 方法进行消息传递，更新节点特征。
        #edge_index:边集 1.[2, 23338] x:线性变换后的特征数组 1.[2664,256] norm:归一化的注意力系数，用于加权求和邻居节点的信息。 1.[23338]
        #返回值是一个二维张量，表示经过图注意力网络处理后每个节点的新特征表示。
        # new = self.propagate(edge_index, x=x, norm=e)
        # print(new.size())
        # return new
        return self.propagate(edge_index, x=x, norm=e)#1.[2664,256]

    def message(self, x_j, norm):
        #x_j：来自邻居节点的消息。
        #norm：归一化后的注意力系数。
        #返回值是加权后的邻居节点特征，用于更新当前节点的特征。
        return norm.view(-1, 1) * x_j
