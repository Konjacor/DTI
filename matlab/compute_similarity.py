import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import time

# 定义网络列表
nets = ['mat_drug_drug', 'mat_drug_disease', 'mat_drug_se',
        'mat_protein_protein', 'mat_protein_disease']

for net in nets:
    start_time = time.time()

    # 构建输入文件路径并加载数据
    input_id = f'../data/{net}.txt'
    M = np.loadtxt(input_id)

    # 计算 Jaccard 距离并转换为相似度矩阵
    sim = 1 - pdist(M, 'jaccard')
    sim = squareform(sim)

    # 调整相似度矩阵
    sim = sim + np.eye(M.shape[0])
    sim = np.nan_to_num(sim, nan=0.0)  # 将所有 NaN 值替换为 0

    # 构建输出文件路径并保存相似度矩阵
    output_id = f'../network/Sim_{net}.txt'
    pd.DataFrame(sim).to_csv(output_id, sep='\t', index=False, header=False)

    end_time = time.time()
    print(f"Processing {net} took {end_time - start_time:.4f} seconds")

# 处理化学相似性和序列相似性
chemical_similarity_path = '../data/Similarity_Matrix_Drugs.txt'
output_chemical_similarity_path = '../network/Sim_mat_Drugs.txt'
M = np.loadtxt(chemical_similarity_path)
pd.DataFrame(M).to_csv(output_chemical_similarity_path, sep='\t', index=False, header=False)

sequence_similarity_path = '../data/Similarity_Matrix_Proteins.txt'
output_sequence_similarity_path = '../network/Sim_mat_Proteins.txt'
M = np.loadtxt(sequence_similarity_path)
pd.DataFrame(M).to_csv(output_sequence_similarity_path, sep='\t', index=False, header=False)