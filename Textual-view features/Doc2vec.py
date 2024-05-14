import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 从CSV文件中读取蛋白质名字和序列
df = pd.read_csv("PBI sequence.csv")

# 获取蛋白质名字和序列
protein_names = df['Protein Name'].tolist()
protein_sequences = df['Protein Sequence'].tolist()

# 将所有序列整合到一个字符串中
all_sequences = ''.join(protein_sequences)

# 获取唯一的字符
unique_characters = list(set(all_sequences))

# 为每个字符创建标记文档
documents = [TaggedDocument(words=[char], tags=[str(i)]) for i, char in enumerate(unique_characters)]

# 定义Doc2Vec模型
model = Doc2Vec(size=64, min_count=1)

# 构建词汇表
model.build_vocab(documents)

# 训练Doc2Vec模型
model.train(documents, total_examples=model.corpus_count, epochs=100)

# 获取每个字符的向量表示
character_vectors = [model.infer_vector([char]) for char in unique_characters]

# 将向量保存为CSV文件
output_df = pd.DataFrame(character_vectors, columns=[f"feature_{i}" for i in range(64)])
output_df.insert(0, 'character', unique_characters)
output_df.to_csv("PBI_doc2vec_vectors.csv", index=False)
