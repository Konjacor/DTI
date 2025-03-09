from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim

from ResidualConnection.ResidualLayer import EnhancedSFGCN
from utils import *
from models import SFGCN

from config import Config

from AutoGCL.augmentation import AugmentationPipeline
from AutoGCL.contrastive_learning import ContrastiveLearning

from torch.optim.lr_scheduler import ReduceLROnPlateau

# from Attention.AttentionLayer import EnhancedSFGCN

# from ResidualConnection.ResidualLayer import EnhancedSFGCN

# MGNN

if __name__ == "__main__":
    config_file = "./config/200dti.ini"#配置文件

    config = Config(config_file)#读取配置文件中的配置
    fold_ROC = []#用来保存实验数据最后计算平均AUROC
    fold_AUPR = []#用来保存实验数据最后计算平均AUPR

    # 初始化 AutoGCL 模块
    augmentation_pipeline = AugmentationPipeline(node_drop_prob=0.05, edge_perturb_prob=0.05)
    contrastive_learning = ContrastiveLearning(temperature=0.03, contrastive_weight=0.05)

    for fold in range(0,5):#读取数据
        # 路径有问题，应该是MGNN_data文件夹而非data文件夹，已经改过来了
        config.structgraph_path = "../MGNN_data/dti/alledg00{}.txt".format(fold + 1)
        config.train_path = "../MGNN_data/dti/train00{}.txt".format(fold + 1)#训练集？
        config.test_path = "../MGNN_data/dti/test00{}.txt".format(fold + 1)#测试集？
        use_seed = not config.no_seed#用不用种子？
        if use_seed:#如果用种子，设置种子？
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        sadj, fadj = load_graph(config)#拿到归一化后的拓扑图邻接矩阵和特征图邻接矩阵。形状都是[2664，2664]。
        # print(sadj)
        # print(fadj)

        # torch.Size([2664, 500]) torch.Size([2664]) torch.Size([2000]) torch.Size([222])
        features, labels, idx_train, idx_test = load_data(config)#拿到节点的特征矩阵（PyTorch FloatTensor 格式）、节点的标签（PyTorch LongTensor 格式）、训练集节点的索引（PyTorch LongTensor 格式）、测试集节点的索引（PyTorch LongTensor 格式）

        #torch.Size([2, 23338])torch.Size([2, 24230])torch.Size([2, 23494])torch.Size([2, 24236])torch.Size([2, 23620])
        asadj = get_adj(sadj)#将拓扑图邻接矩阵（adj）转换为图的边索引格式（edge_index），以便用于图神经网络（如 PyTorch Geometric）。
        #torch.Size([2, 7206])torch.Size([2, 7206])torch.Size([2, 7206])torch.Size([2, 7206])torch.Size([2, 7206])
        afadj = get_adj(fadj)#将特征图邻接矩阵（adj）转换为图的边索引格式（edge_index），以便用于图神经网络（如 PyTorch Geometric）。

        model = SFGCN(nfeat=config.fdim,#特征数500
                      nhid1=config.nhid1,#隐藏层1特征数256
                      nhid2=config.nhid2,#隐藏层2特征数64
                      nclass=config.class_num,#2
                      n=config.n,#2664
                      dropout=config.dropout#在特定的训练迭代中要忽略的神经元比例0.2，防止过拟合
                      )#定义了一个类名为 SFGCN 的神经网络模型，继承自 PyTorch 的 nn.Module 类。它结合了图注意力网络（GAT）和图卷积网络（GCN），并通过注意力机制和多层感知机（MLP）实现对节点特征的处理和分类。

        # model = EnhancedSFGCN(nfeat=config.fdim,
        #               nhid1=config.nhid1,
        #               nhid2=config.nhid2,
        #               nclass=config.class_num,
        #               n=config.n,
        #               dropout=config.dropout)

        features = features#torch.Size([2664, 500])
        sadj = sadj#[2664，2664]
        fadj = fadj#[2664，2664]
        labels = labels#torch.Size([2664])
        idx_train = idx_train#torch.Size([2000])
        idx_test = idx_test#torch.Size([222])
        #创建一个 Adam 优化器实例，并将其配置为使用指定的学习率 (lr) 和权重衰减 (weight_decay) 来优化给定模型的所有可训练参数。
        #model.parameters()作用：返回模型中所有可训练参数的迭代器。这些参数是在模型定义时使用 nn.Parameter 或通过 nn.Module 的子类（如卷积层、全连接层等）自动注册的。优化器需要知道哪些参数需要被更新，因此传入 model.parameters() 让优化器知道要优化哪些参数。
        #权重衰减 (Weight Decay)：也称为 L2 正则化，是一种防止过拟合的技术。通过在损失函数中添加与权重大小成正比的惩罚项来限制模型的复杂度。
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)#使用 Adam 优化器对模型参数进行优化
        roc = []

        pr = []
        acc = []

        # 使用学习率调度器在训练过程中动态调整学习率。5个epoch后如果特定参数还没有优化，则将学习率下降为以前的0.1倍
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # def train(model, epochs):
        #     model.train()
        #     optimizer.zero_grad()
        #     output = model(features, sadj, fadj, asadj, afadj)#前向传播
        #     loss = F.nll_loss(output[idx_train], labels[idx_train])
        #
        #     loss.backward()
        #     optimizer.step()
        #     c, d, p = main_test(model)
        #     print("this is the", epochs + 1,
        #           "epochs, ROC is {:.4f},and AUPR is {:.4f} test set accuray is {:.4f},loss is {:.4f} ".format(c, d, p,
        #                                                                                                        loss))

        def train(model, epochs):
            model.train()#设置模型为训练模式
            optimizer.zero_grad()#清空优化器中的梯度，防止梯度积累影响当前批次的更新

            # 前向传播得到模型输出
            #features:torch.Size([2664, 500]) sadj:[2664，2664] fadj:[2664，2664]
            #asadj:torch.Size([2, 23338])torch.Size([2, 24230])torch.Size([2, 23494])torch.Size([2, 24236])torch.Size([2, 23620])
            #afadj:torch.Size([2, 7206])torch.Size([2, 7206])torch.Size([2, 7206])torch.Size([2, 7206])torch.Size([2, 7206])
            output = model(features, sadj, fadj, asadj, afadj)#1.[2664,2]

            # 监督损失，使用负对数似然损失函数 (F.nll_loss) 计算监督损失，仅针对训练集的数据点（idx_train）。
            supervised_loss = F.nll_loss(output[idx_train], labels[idx_train])

            # 数据增广
            aug_sadj = augmentation_pipeline(asadj)
            aug_fadj = augmentation_pipeline(afadj)

            # 增广视图的前向传播，使用增强后的邻接矩阵再次进行前向传播，获取增广视图下的输出。
            aug_output = model(features, sadj, fadj, aug_sadj, aug_fadj)

            # 计算对比损失，旨在使原始视图和增广视图下的表示尽可能相似。
            contrastive_loss = contrastive_learning.compute_contrastive_loss(output, aug_output)

            # 总损失
            loss = supervised_loss + contrastive_loss

            #反向传播更新参数
            loss.backward()
            optimizer.step()
            scheduler.step(loss)#动态调整学习率

            c, d, p = main_test(model)#ROC、AUPR和准确率
            print("this is the", epochs + 1,
                  "epochs, ROC is {:.4f},and AUPR is {:.4f} test set accuray is {:.4f},loss is {:.4f}".format(c, d, p,
                                                                                                              loss))

        #用于评估一个给定模型在测试集上的性能。具体来说，它会将模型设置为评估模式，进行前向传播以获取模型的输出，并计算一些评价指标（如 ROC、AUPR 和准确率）。
        def main_test(model):
            #将模型设置为评估模式。在评估模式下，模型的行为与训练模式不同，例如，某些层（如 Dropout 或 BatchNorm）会有不同的行为。
            model.eval()#将模型设置为评估模式
            output = model(features, sadj, fadj, asadj, afadj)#前向传播得到模型输出1.[2664,2]
            c, d = RocAndAupr(output[idx_test], labels[idx_test])#计算ROC和AUPR值
            acc_test = accuracy(output[idx_test], labels[idx_test])#计算准确率
            roc.append(c)#将ROC值添加到roc列表中
            pr.append(d)#将AUPR值添加到pr列表中
            acc.append(acc_test)#将准确率添加到acc列表中
            return c, d, acc_test#返回ROC、AUPR和准确率


        acc_max = 0
        epoch_max = 0
        roc_max = 0
        pr_max = 0

        for epoch in range(config.epochs):
            train(model, epoch)
            if acc_max < acc[epoch]:
                acc_max = acc[epoch]
            if roc_max < roc[epoch]:
                roc_max = roc[epoch]
            if pr_max < pr[epoch]:
                pr_max = pr[epoch]
            if epoch + 1 == config.epochs:
                fold_ROC.append(roc_max)
                fold_AUPR.append(pr_max)
                print(
                    "this is {} fold ,the max ROC is {:.4f},and max AUPR is {:.4f} test set max  accuray is {:.4f} , ".format(fold,roc_max,
                                                                                                             pr_max,
                                                                                                             acc_max))
    print("average AUROC is {:.4} , average AUPR is {:.4}".format(sum(fold_ROC) / len(fold_ROC),
                                                                  sum(fold_AUPR) / len(fold_AUPR)))
