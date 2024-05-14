import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 函数用于计算蛋白质序列的 CTD 特征
def compute_ctd_features(sequence):
    # 定义氨基酸的属性
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    num_aa = len(amino_acids)

    # 初始化 CTD 特征向量
    ctd_features = np.zeros((440,))

    # 计算氨基酸组成（Composition）
    for i, aa in enumerate(amino_acids):
        ctd_features[i] = sequence.count(aa) / len(sequence)

    # 计算氨基酸转换（Transition）
    idx = 20
    for i in range(num_aa):
        for j in range(num_aa):
            ctd_features[idx] = abs(sequence.count(amino_acids[i]) - sequence.count(amino_acids[j]))
            idx += 1

    # 计算氨基酸分布（Distribution）
    for i, aa in enumerate(amino_acids):
        ctd_features[idx] = sequence.count(aa)
        idx += 1

    return ctd_features

# 读取CSV文件
data = pd.read_csv('PBI sequence.csv')

# 对每个蛋白质序列计算CTD特征
ctd_features = []
for sequence in data['Sequence']:
    ctd = compute_ctd_features(sequence)
    ctd_features.append(ctd)

# 使用PCA将CTD特征降维为64维
pca = PCA(n_components=64)
ctd_features_pca = pca.fit_transform(ctd_features)

# 将结果转换为DataFrame
ctd_df = pd.DataFrame(ctd_features_pca, columns=['Feature_' + str(i) for i in range(1, 65)])

# 将蛋白质名称与特征合并
result_df = pd.concat([data['Protein'], ctd_df], axis=1)

# 保存结果到CSV文件
result_df.to_csv('PBI_ctd_features.csv', index=False)
