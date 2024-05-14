import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取包含余弦相似度的 CSV 文件
df_similarity = pd.read_csv('cosine_similarity_results.csv')

# 重新排列数据以便于绘制热图
df_heatmap = df_similarity.pivot('Protein_A', 'Protein_B', 'Similarity')

# 绘制热图
plt.figure(figsize=(15, 8))
sns.heatmap(df_heatmap, cmap=plt.get_cmap('Reds'), fmt=".2f", linewidths='1', linecolor='black')
plt.title('Cosine Similarity Heatmap between Protein A and Protein B')
plt.xlabel('Known Phages')
plt.ylabel('Unknown Phages')

# 保存图片
plt.savefig('cosine_similarity_heatmap.png')

plt.show()
