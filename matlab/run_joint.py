import numpy as np
import os
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import norm as sparse_norm
import time


def diffusionRWR(A, maxiter, restartProb):
    n = A.shape[0]  # n表示矩阵A的行数

    # Add self-edge to isolated nodes
    A += np.diag((A.sum(axis=1) == 0).A1)

    # Normalize the adjacency matrix
    P = A.multiply(1 / A.sum(axis=1).A1[:, np.newaxis])

    # Personalized PageRank
    restart = identity(n, format='csr')
    Q = identity(n, format='csr')

    for i in range(maxiter):
        Q_new = (1 - restartProb) * P.dot(Q) + restartProb * restart
        delta = sparse_norm(Q - Q_new, 'fro')
        Q = Q_new.copy()
        if delta < 1e-6:
            break

    return Q.toarray()


def joint(networks, rsp, maxiter):
    Q = None
    for net_name in networks:
        file_path = os.path.join('../network', f'{net_name}.txt')
        net = np.loadtxt(file_path)
        net_sparse = csr_matrix(net)
        tQ = diffusionRWR(net_sparse, maxiter, rsp)

        if Q is None:
            Q = tQ
        else:
            Q = np.hstack((Q, tQ))

    nnode = Q.shape[0]
    alpha = 1 / nnode
    Q = np.log(Q + alpha) - np.log(alpha)
    return Q


# 主程序
maxiter = 20
restartProb = 0.50

drug_nets = ['Sim_mat_drug_drug', 'Sim_mat_drug_disease', 'Sim_mat_drug_se', 'Sim_mat_Drugs']
protein_nets = ['Sim_mat_protein_protein', 'Sim_mat_protein_disease', 'Sim_mat_Proteins']

# 处理 drug_nets
start_time = time.time()
X = joint(drug_nets, restartProb, maxiter)
end_time = time.time()
print(f"Processing drug_nets took {end_time - start_time:.4f} seconds")

# 处理 protein_nets
start_time = time.time()
Y = joint(protein_nets, restartProb, maxiter)
end_time = time.time()
print(f"Processing protein_nets took {end_time - start_time:.4f} seconds")

# 保存结果,两个结果都进行了归一化处理
np.savetxt('../feature/drug_vector.txt', X, delimiter='\t')
np.savetxt('../feature/protein_vector.txt', Y, delimiter='\t')