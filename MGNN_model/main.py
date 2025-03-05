from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models import SFGCN

from config import Config

from AutoGCL.augmentation import AugmentationPipeline
from AutoGCL.contrastive_learning import ContrastiveLearning

# MGNN

if __name__ == "__main__":
    config_file = "./config/200dti.ini"#配置文件

    config = Config(config_file)#读取配置文件中的配置
    fold_ROC = []#
    fold_AUPR = []#

    # 初始化 AutoGCL 模块
    augmentation_pipeline = AugmentationPipeline(node_drop_prob=0.1, edge_perturb_prob=0.1)
    contrastive_learning = ContrastiveLearning(temperature=0.1, contrastive_weight=0.1)

    for fold in range(0,5):#读取数据
        # 路径有问题，应该是MGNN_data文件夹而非data文件夹，已经改过来了
        config.structgraph_path = "../MGNN_data/dti/alledg00{}.txt".format(fold + 1)
        config.train_path = "../MGNN_data/dti/train00{}.txt".format(fold + 1)#训练集？
        config.test_path = "../MGNN_data/dti/test00{}.txt".format(fold + 1)#测试集？
        use_seed = not config.no_seed#用不用种子？
        if use_seed:#如果用种子，设置种子？
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        sadj, fadj = load_graph(config)#拿到归一化后的拓扑图邻接矩阵和特征图邻接矩阵。
        features, labels, idx_train, idx_test = load_data(config)#拿到节点的特征矩阵（PyTorch FloatTensor 格式）、节点的标签（PyTorch LongTensor 格式）、训练集节点的索引（PyTorch LongTensor 格式）、测试集节点的索引（PyTorch LongTensor 格式）

        asadj = get_adj(sadj)#将拓扑图邻接矩阵（adj）转换为图的边索引格式（edge_index），以便用于图神经网络（如 PyTorch Geometric）。
        afadj = get_adj(fadj)#将特征图邻接矩阵（adj）转换为图的边索引格式（edge_index），以便用于图神经网络（如 PyTorch Geometric）。

        model = SFGCN(nfeat=config.fdim,
                      nhid1=config.nhid1,
                      nhid2=config.nhid2,
                      nclass=config.class_num,
                      n=config.n,
                      dropout=config.dropout)#定义了一个类名为 SFGCN 的神经网络模型，继承自 PyTorch 的 nn.Module 类。它结合了图注意力网络（GAT）和图卷积网络（GCN），并通过注意力机制和多层感知机（MLP）实现对节点特征的处理和分类。


        features = features
        sadj = sadj
        fadj = fadj
        labels = labels
        idx_train = idx_train
        idx_test = idx_test
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)#使用 Adam 优化器对模型参数进行优化
        roc = []

        pr = []
        acc = []


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
            model.train()
            optimizer.zero_grad()

            # 前向传播
            output = model(features, sadj, fadj, asadj, afadj)

            # 监督损失
            supervised_loss = F.nll_loss(output[idx_train], labels[idx_train])

            # 数据增广
            aug_sadj = augmentation_pipeline(asadj)
            aug_fadj = augmentation_pipeline(afadj)

            # 增广视图的前向传播
            aug_output = model(features, sadj, fadj, aug_sadj, aug_fadj)

            # 对比损失
            contrastive_loss = contrastive_learning.compute_contrastive_loss(output, aug_output)

            # 总损失
            loss = supervised_loss + contrastive_loss

            loss.backward()
            optimizer.step()

            c, d, p = main_test(model)
            print("this is the", epochs + 1,
                  "epochs, ROC is {:.4f},and AUPR is {:.4f} test set accuray is {:.4f},loss is {:.4f}".format(c, d, p,
                                                                                                              loss))

        def main_test(model):
            model.eval()
            output = model(features, sadj, fadj, asadj, afadj)
            c, d = RocAndAupr(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            roc.append(c)
            pr.append(d)
            acc.append(acc_test)
            return c, d, acc_test


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
