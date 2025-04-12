import numpy as np
from DAE import DAE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# drug_train = np.loadtxt(r"../feature/drug_vector.txt")
# protein_train = np.loadtxt(r"../feature/protein_vector.txt")
drug_train1 = np.loadtxt(r"../feature/drug_vector1.txt")
protein_train1 = np.loadtxt(r"../feature/protein_vector1.txt")
drug_train2 = np.loadtxt(r"../feature/drug_embeddings_normalized.txt")
protein_train2 = np.loadtxt(r"../feature/protein_embeddings_normalized.txt")

# print(drug_train.shape)
# print(protein_train.shape)
print(drug_train1.shape)
print(protein_train1.shape)
print(drug_train2.shape)
print(protein_train2.shape)

# drug_size=drug_train.shape[1]
# protein_size=protein_train.shape[1]
drug_size1=drug_train1.shape[1]
protein_size1=protein_train1.shape[1]
drug_size2=drug_train2.shape[1]
protein_size2=protein_train2.shape[1]

# print(drug_size,protein_size)
print(drug_size1,protein_size1)
print(drug_size2,protein_size2)

# drug_train = np.concatenate((drug_train,drug_train1),1)
# protein_train = np.concatenate((protein_train,protein_train1),1)
drug_train = np.concatenate((drug_train1,drug_train2),1)
protein_train = np.concatenate((protein_train1,protein_train2),1)
drug_size=drug_train.shape[1]
protein_size=protein_train.shape[1]
print(drug_size,protein_size)

# training_epochs之前是20
data1=DAE(drug_train,drug_size,100,16,1,100,[100])
# # 假设X是原始数据（形状为[N_samples, 160]）
# X = drug_train
#
# # 检测是否存在NaN
# has_nan = np.isnan(X).any()
#
# print(f"矩阵中是否存在NaN值: {has_nan}")  # 输出: True
# print(X)
# # 1. 标准化数据
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print(X_scaled)
#
# # 2. 应用PCA
# pca = PCA(n_components=100)
# X_compressed = pca.fit_transform(X_scaled)
#
# print(f"压缩后的维度：{X_compressed.shape}")  # 输出：(N_samples, 100)
np.savetxt('../DAE/drug_dae_d100.txt',data1)

# training_epochs之前是20
data2=DAE(protein_train,protein_size,100,32,1,400,[400])
np.savetxt('../DAE/protein_dae_d400.txt',data2)