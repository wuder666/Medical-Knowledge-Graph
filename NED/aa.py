import pandas as pd
import numpy as np
import os
import collections
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 获取当前工作目录路径
bast_path = os.getcwd()

# TODO：将entity_list.csv中已知实体的名称导入分词词典
# 读取实体列表文件（entity_list.csv），包含实体名称和描述信息
entity_data = pd.read_csv(os.path.join(bast_path, 'data/entity_disambiguation/entity_list.csv'), encoding='utf-8')
print(f'entity_data--》{entity_data.head()}')

# TODO：对每句句子识别并匹配实体
# 读取包含待处理句子的文件（valid_data.csv）
valid_data = pd.read_csv(os.path.join(bast_path, 'data/entity_disambiguation/valid_data.csv'), encoding='gb18030')
print(f'valid_data--》{valid_data.head()}')

# 将实体名称拼接成一个长字符串，并用'|'分隔，用于统计实体名称的出现次数
s = ''
keyword_list = []
for i in entity_data['entity_name'].values.tolist():
    s += i + '|'

# 统计实体名称在字符串中的出现次数，如果某个名称出现次数超过一次，则将其加入keyword_list（关键词列表）
for k, v in collections.Counter(s.split('|')).items():
    if v > 1:
        keyword_list.append(k)

# 生成TF-IDF矩阵

# 对实体的描述信息进行分词，将每个实体描述分词后的结果存入train_sentence列表中
train_sentence = []
for i in entity_data['desc'].values:
    train_sentence.append(' '.join(jieba.cut(i)))
print(len(train_sentence))

# 初始化TF-IDF向量化工具
vectorizer = TfidfVectorizer()

# 将实体描述信息转换为TF-IDF特征矩阵
X = vectorizer.fit_transform(train_sentence)
print(X)
print(X.toarray().shape)

# 定义获取实体ID的函数，根据给定的句子计算其与实体描述的TF-IDF余弦相似度，返回最相似的实体ID
def get_entityid(sentence):
    id_start = 1001  # 假设实体ID从1001开始
    a_list = [' '.join(jieba.cut(sentence))]  # 对输入句子分词
    print(f'a_list--》{a_list}')
    print(vectorizer.transform(a_list))  # 将句子转换为TF-IDF特征
    print(cosine_similarity(vectorizer.transform(a_list), X))  # 计算句子与所有实体描述的余弦相似度
    res = cosine_similarity(vectorizer.transform(a_list), X)[0]  # 获取相似度结果
    top_idx = np.argsort(res)[-1]  # 获取最相似的实体在TF-IDF矩阵中的索引
    print(f'np.argsort(res)==>{np.argsort(res)}')
    return id_start + top_idx  # 返回实体ID

# TODO：将计算结果存入文件
print(f'keyword_list--》{keyword_list}')

# 初始化行计数器和结果列表
row = 0
result_data = []
neighbor_sentence = ''

# 遍历valid_data中的每一个句子，处理其中的关键词
for sentence in valid_data['sentence']:
    print(f'sentence--》{sentence}')
    res = [row]  # 初始化结果列表，首先添加当前行号
    for keyword in keyword_list:
        if keyword in sentence:  # 如果句子中包含关键词
            print(f'keyword--》{keyword}')
            k_len = len(keyword)  # 计算关键词的长度
            print(f'k_len--》{k_len}')
            ss = ''
            for i in range(len(sentence) - k_len + 1):
                if sentence[i:i+k_len] == keyword:  # 如果在句子中找到关键词
                    s = str(i) + '-' + str(i + k_len) + ':'  # 获取关键词在句子中的位置（如"0-5"）
                    print(f's-->{s}')
                    # 获取包含关键词的邻近句子，用于计算实体相似度
                    if i > 10 and i + k_len < len(sentence) - 9:
                        neighbor_sentence = sentence[i-10:i+k_len+9]
                    elif i < 10:
                        neighbor_sentence = sentence[:20]
                    elif i + k_len > len(sentence) - 9:
                        neighbor_sentence = sentence[-20:]

                    # 调用get_entityid函数，获取与邻近句子最相似的实体ID
                    s += str(get_entityid(neighbor_sentence))
                    ss += s + '|'  # 将位置和实体ID拼接成字符串
            res.append(ss[:-1])  # 将处理结果加入到当前行的结果中
        break
    result_data.append(res)  # 将当前句子的处理结果加入结果列表
    row += 1  # 行计数器加1

# 将结果保存为CSV文件，文件路径为'entity_disambiguation_submit.csv'
pd.DataFrame(result_data).to_csv('entity_disambiguation_submit.csv', index=False)
