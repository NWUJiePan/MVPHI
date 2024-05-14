import numpy as np
import itertools


# 将蛋白质序列转换为 k-mer 特征向量
def protein_to_kmer_feature(protein_sequence, k=4, feature_dim=64):
    # 创建一个包含所有可能 k-mer 的列表
    all_kmers = [''.join(x) for x in itertools.product('ACDEFGHIKLMNPQRSTVWY', repeat=k)]
    num_kmers = len(all_kmers)

    # 初始化特征向量
    feature_vector = np.zeros(num_kmers)

    # 遍历蛋白质序列，并计算每个 k-mer 的出现次数
    for i in range(len(protein_sequence) - k + 1):
        kmer = protein_sequence[i:i + k]
        if kmer in all_kmers:
            feature_vector[all_kmers.index(kmer)] += 1

    # 将特征向量归一化
    feature_vector /= np.linalg.norm(feature_vector)

    # 如果特征维度不为 64，则进行维度调整
    if len(feature_vector) != feature_dim:
        feature_vector = resize_feature_vector(feature_vector, feature_dim)

    return feature_vector


# 调整特征向量的维度为 64
def resize_feature_vector(feature_vector, target_dim=64):
    current_dim = len(feature_vector)
    if current_dim > target_dim:
        # 如果特征维度大于目标维度，则截取前 target_dim 个维度
        return feature_vector[:target_dim]
    else:
        # 如果特征维度小于目标维度，则在末尾填充 0
        return np.pad(feature_vector, (0, target_dim - current_dim), 'constant')


# 示例蛋白质序列
protein_sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG"

# 提取 k-mer 特征
kmer_feature = protein_to_kmer_feature(protein_sequence)

print("K-mer Feature Vector:")
print(kmer_feature)
print("Feature Vector Dimension:", len(kmer_feature))
