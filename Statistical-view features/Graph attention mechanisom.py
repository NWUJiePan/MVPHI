import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 加载第一个 CSV 文件的特征向量和标签
df1 = pd.read_csv("PBI_4kmer_feature.csv")  # 请替换为你的文件路径
features1 = df1.iloc[:, 1:].values  # 第一列是标签，剩余列是特征向量
labels1 = df1.iloc[:, 0].values

# 加载第二个 CSV 文件的特征向量和标签
df2 = pd.read_csv("PBI_ctd_features.csv")  # 请替换为你的文件路径
features2 = df2.iloc[:, 1:].values  # 第一列是标签，剩余列是特征向量
labels2 = df2.iloc[:, 0].values

# 将特征和标签转换为 PyTorch 张量
features1_tensor = torch.tensor(features1, dtype=torch.float32)
features2_tensor = torch.tensor(features2, dtype=torch.float32)
labels1_tensor = torch.tensor(labels1, dtype=torch.long)
labels2_tensor = torch.tensor(labels2, dtype=torch.long)

# 定义图注意力模型
class GraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphAttention, self).__init__()
        self.attention = nn.Linear(input_dim * 2, output_dim)  # 输入维度应为特征向量的维度的两倍

    def forward(self, x1, x2):
        # 在这里实现图注意力的操作
        # 这里只是一个示例，你需要根据你的需求来实现图注意力的具体操作
        output = torch.cat((x1, x2), dim=1)  # 将两个特征向量拼接在一起
        output = self.attention(output)
        return output

# 初始化图注意力模型
input_dim = features1_tensor.shape[1]  # 输入特征的维度为特征向量的维度
output_dim = 64  # 你可以根据需要调整输出特征的维度
graph_attention_model = GraphAttention(input_dim, output_dim)

# 将特征向量输入到图注意力模型中进行特征融合
with torch.no_grad():
    fused_features = graph_attention_model(features1_tensor, features2_tensor).numpy()

# 将融合后的特征向量和标签合并为 DataFrame
fused_df = pd.DataFrame(np.hstack((labels1.reshape(-1, 1), fused_features)), columns=["label"] + [f"feature_{i}" for i in range(fused_features.shape[1])])

# 保存融合后的特征向量为 CSV 文件
fused_df.to_csv("PBI_统计学_features.csv", index=False)
