import numpy as np
import pandas as pd
import os
from sklearn_extra.cluster import KMedoids

def cluster_kmedoids(features, n_clusters):
    """使用k-medoids算法进行聚类"""
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    kmedoids.fit(features)
    return kmedoids.labels_

#n_drug_clusters和n_protein_clusters是簇的数量，取根号下n/2时效果最好，NEGNO是最后返回的负样本数量，之前默认是取和正样本一样的数量。
def generate_optimized_neg_samples(mat_drug_protein, drug_features, protein_features,
                                   NEGNO,n_drug_clusters=19, n_protein_clusters=28):
    """优化后的负样本生成方法"""
    # ------------------------------------------
    # Step 1. 聚类药物和蛋白质
    # ------------------------------------------
    # 药物聚类 (708个药物)
    drug_labels = cluster_kmedoids(drug_features, n_drug_clusters)  # shape: (708,)
    # 蛋白质聚类 (1512个蛋白质)
    protein_labels = cluster_kmedoids(protein_features, n_protein_clusters)  # shape: (1512,)
    print("step 1 done")
    # ------------------------------------------
    # Step 2. 构建NEG集合
    # ------------------------------------------
    NEG = []
    # 遍历所有簇对
    for dc in range(n_drug_clusters):
        for pc in range(n_protein_clusters):
            # 获取当前药物簇和蛋白质簇的索引
            drug_indices = np.where(drug_labels == dc)[0]  # 属于dc簇的药物索引
            protein_indices = np.where(protein_labels == pc)[0]  # 属于pc簇的蛋白质索引

            # 检查簇对之间是否存在相互作用
            interaction_exists = np.any(mat_drug_protein[np.ix_(drug_indices, protein_indices)])

            # 如果簇对无相互作用，则加入NEG
            if not interaction_exists:
                # 生成所有可能的药物-蛋白质对
                pairs = [(d, p) for d in drug_indices for p in protein_indices]
                NEG.extend(pairs)

    NEG = list(set(NEG))  # 去重
    print("step 2 done")
    # ------------------------------------------
    # Step 3. 构建RNEG子集
    # ------------------------------------------
    if len(NEG) <= NEGNO:
        RNEG = NEG
    else:
        # print(1111111)
        RNEG = []
        # 计算每个簇对的采样比例
        total_pairs = len(NEG)
        for dc in range(n_drug_clusters):
            for pc in range(n_protein_clusters):
                drug_indices = np.where(drug_labels == dc)[0]
                protein_indices = np.where(protein_labels == pc)[0]

                # 计算当前簇对的潜在负样本数k
                k = len(drug_indices) * len(protein_indices)
                if k == 0:
                    continue

                # 计算需要采样的数量r
                r = int((k * NEGNO) / total_pairs)

                # 从当前簇对中采样r个负样本
                if r > 0:
                    sampled_drugs = np.random.choice(drug_indices, size=min(r, len(drug_indices)), replace=False)
                    sampled_proteins = np.random.choice(protein_indices, size=min(r, len(protein_indices)), replace=False)
                    RNEG.extend([(d, p) for d in sampled_drugs for p in sampled_proteins])

        # 确保最终数量不超过NEGNO
        RNEG = list(set(RNEG))[:NEGNO]
    print("step 3 done")
    return RNEG


def load_data(file_path):
    """加载数据文件"""
    return np.loadtxt(file_path)


def load_features(drug_feature_path, protein_feature_path):
    """加载药物和蛋白质的特征向量"""
    drug_features = np.loadtxt(drug_feature_path)
    protein_features = np.loadtxt(protein_feature_path)
    return drug_features, protein_features

def generate_negative_samples(mat_drug_protein, num_samples):
    """生成负样本"""
    neg_samples = []
    num_drugs, num_proteins = mat_drug_protein.shape
    # 随机选取负样本，直到达到所需数量
    while len(neg_samples) < num_samples:
        drug_idx = np.random.randint(0, num_drugs)
        protein_idx = np.random.randint(0, num_proteins)
        if mat_drug_protein[drug_idx, protein_idx] == 0:
            neg_samples.append((drug_idx, protein_idx))
    return neg_samples

def create_dti_feature(mat_drug_protein_path, drug_features, protein_features, output_path):
    """根据相互作用矩阵创建DPP特征向量"""
    mat_drug_protein = load_data(mat_drug_protein_path)
    print(mat_drug_protein.shape)#

    # 获取药物和蛋白质的数量
    num_drugs, num_proteins = mat_drug_protein.shape

    # 初始化DPP特征矩阵
    dti_features = []

    # 遍历相互作用矩阵，收集正样本
    positive_samples = []
    for drug_idx in range(num_drugs):
        for protein_idx in range(num_proteins):
            if mat_drug_protein[drug_idx, protein_idx] == 1:
                # 保存药物和蛋白质的坐标
                positive_samples.append((drug_idx, protein_idx))

    # 生成与正样本等量的负样本
    # negative_samples = generate_negative_samples(mat_drug_protein, len(positive_samples))
    negative_samples = generate_optimized_neg_samples(mat_drug_protein,drug_features,protein_features,len(positive_samples))

    print("positive_samples:",positive_samples)
    print("negative_samples:",negative_samples)
    all_samples = positive_samples + negative_samples
    print("all_samples:",all_samples)

    # 拼接正样本和负样本的特征向量
    for drug_idx, protein_idx in positive_samples:
        # print(drug_idx,protein_idx)
        combined_feature = np.concatenate([drug_features[drug_idx], protein_features[protein_idx]])
        dti_features.append(combined_feature)

    for drug_idx, protein_idx in negative_samples:
        combined_feature = np.concatenate([drug_features[drug_idx], protein_features[protein_idx]])
        dti_features.append(combined_feature)

    # 转换为NumPy数组
    all_samples = np.array(all_samples)
    print(all_samples.shape)#(2664,2)
    dti_features = np.array(dti_features)
    print(dti_features.shape)#(2664,500)


    # 保存结果
    dpp_path = "./dpp.txt"
    np.savetxt(dpp_path, all_samples,fmt='%d')
    print(f"所有DPP对的坐标已保存到 {dpp_path}")
    np.savetxt(output_path, dti_features,fmt='%.15f')
    print(f"DPP特征向量已保存到 {output_path}")


if __name__ == "__main__":
    # 文件路径
    mat_drug_protein_path = "../data/mat_drug_protein.txt"
    drug_feature_path = "./drug_dae_d100.txt"
    protein_feature_path = "./protein_dae_d400.txt"
    output_path = "./dti.feature"
    # output_path = "../MGNN_data/dti/dti.feature"

    # 加载特征
    drug_features, protein_features = load_features(drug_feature_path, protein_feature_path)

    print(drug_features.shape)#
    print(protein_features.shape)#

    # 创建DPP特征向量
    create_dti_feature(mat_drug_protein_path, drug_features, protein_features, output_path)