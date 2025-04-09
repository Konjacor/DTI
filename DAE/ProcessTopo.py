from collections import defaultdict
import random
import os


def generate_edges(input_file, output_file):
    # 读取 DPP 数据并记录索引
    dpps = []
    with open(input_file, "r") as f:
        for idx, line in enumerate(f):
            drug, target = map(int, line.strip().split())
            dpps.append((drug, target, idx))  # (drug, target, dpp_index)

    # 构建倒排索引：药物/靶标 -> [dpp_index]
    drug_to_indices = defaultdict(list)
    target_to_indices = defaultdict(list)
    for drug, target, idx in dpps:
        drug_to_indices[drug].append(idx)
        target_to_indices[target].append(idx)

    # 生成边集（自动去重）
    edges = set()

    # 处理共享药物的边
    for indices in drug_to_indices.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                u, v = indices[i], indices[j]
                edges.add((min(u, v), max(u, v)))  # 确保 u < v 避免重复

    # 处理共享靶标的边
    for indices in target_to_indices.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                u, v = indices[i], indices[j]
                edges.add((min(u, v), max(u, v)))

    # 写入文件（按升序排序）
    with open(output_file, "w") as f:
        for u, v in sorted(edges):
            f.write(f"{u}\t{v}\n")
    print("DPP对的拓扑图的边集已被保存到",output_file)

#5折交叉验证
def split_edges_for_cv(input_file, output_dir, n_folds=5):
    # 读取原始边集并去重（确保原始数据本身无重复）
    with open(input_file, "r") as f:
        edges = list({line.strip() for line in f})  # 去重处理

    total = len(edges)
    sample_size = total // 2  # 每份抽取一半的边（向下取整）

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成五份子集文件
    for fold in range(1, n_folds + 1):
        # 从原始数据中随机抽取不重复的边（允许跨文件重复）
        sampled_edges = random.sample(edges, k=sample_size)

        # 保存文件
        output_file = os.path.join(output_dir, f"alledg00{fold}.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(sampled_edges))
        print("部分DPP拓扑图边集数据已被保存到",f"{output_dir}alledg00{fold}.txt")

if __name__ == "__main__":
    generate_edges("./dpp.txt", "./alledge.txt")

    # 输入文件和输出目录
    split_edges_for_cv(
        input_file="./alledge.txt",
        output_dir="./topo/",
        n_folds=5
    )