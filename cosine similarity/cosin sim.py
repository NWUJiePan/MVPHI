from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 读取CSV文件
df = pd.read_csv('interactions.csv')

# 提取蛋白A和蛋白B的序列
sequences_A = df['Sequences_A'].tolist()
sequences_B = df['Sequences_B'].tolist()

# 合并序列为列表
all_sequences = sequences_A + sequences_B

# 使用TF-IDF向量化器
vectorizer = TfidfVectorizer(analyzer='char', lowercase=False)
tfidf_matrix = vectorizer.fit_transform(all_sequences)

# 计算余弦相似度
cosine_similarities = cosine_similarity(tfidf_matrix[:len(sequences_A)], tfidf_matrix[len(sequences_A):])

# 创建一个 DataFrame 存储相似度结果
results_df = pd.DataFrame({'Protein_A': sequences_A, 'Protein_B': sequences_B, 'Similarity': cosine_similarities[:, 0]})

# 导出结果到 CSV 文件
results_df.to_csv('similarity_results.csv', index=False)

print("相似度结果已保存到 similarity_results.csv 文件中。")
