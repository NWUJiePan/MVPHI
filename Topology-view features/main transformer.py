import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# 读取CSV文件，假设文件中有两列节点，分别为 source 和 target
df = pd.read_csv("RBP pairs.csv")

# 构建节点列表，合并 source 和 target 列并去重
nodes = np.unique(df[['source', 'target']].values)

# 构建节点索引字典
node_to_index = {node: index for index, node in enumerate(nodes)}
index_to_node = {index: node for index, node in enumerate(nodes)}

# 构建邻接矩阵
num_nodes = len(nodes)
adj_matrix = torch.zeros(num_nodes, num_nodes)
for _, row in df.iterrows():
    source_idx = node_to_index[row['source']]
    target_idx = node_to_index[row['target']]
    adj_matrix[source_idx][target_idx] = 1
    adj_matrix[target_idx][source_idx] = 1


# 定义图转换器网络模型
class GraphTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(GraphTransformer, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(input_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, X, adj_matrix):
        # 将节点表示与其自身的注意力权重进行加权求和
        for i in range(self.num_layers):
            # 使用自注意力机制计算新的表示
            X, _ = self.attention_layers[i](X, X, X, attn_mask=adj_matrix)
        return X


# 节点特征维度
input_dim = 64
# 注意力头数
num_heads = 3
# Transformer层数
num_layers = 2

# 定义图转换器模型
gtn_model = GraphTransformer(input_dim, num_heads, num_layers)

# 假设每个节点的特征向量是随机初始化的，可以根据需要修改初始化方式
X = torch.randn(num_nodes, input_dim)

# 对节点特征进行编码
output = gtn_model(X, adj_matrix)

# 将节点名称与向量组合成 DataFrame
output_df = pd.DataFrame(output.detach().numpy(), columns=[f"feature_{i}" for i in range(input_dim)])
output_df['node_name'] = nodes

# 调整列顺序，使得节点名称在第一列
cols = output_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
output_df = output_df[cols]

# 将输出的节点向量保存到CSV文件中
output_df.to_csv("PBI_GT_features.csv", index=False)
