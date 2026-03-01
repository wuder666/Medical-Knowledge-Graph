# coding:utf-8
from Bilstm_Attention_RE.model.bilstm_attention import *
from utils.data_loader import *
from utils.process import *
import torch
from tqdm import tqdm

# 导入配置文件
conf = Config()

# 获取或定义模型参数
word2id, id2word = get_word_id(conf.train_path)
vocab_size = len(word2id)
pos_size = 143
tag_size = len(relation2id)
print(tag_size)

# 获取id2relation的映射
id2relation = {int(value): key for key, value in relation2id.items()}

# 加载数据集
_, test_iter = get_loader_data()

# 实例化Bilstm+attention模型
ba_model = BiLSTM_ATT(conf, vocab_size, pos_size, tag_size).to(conf.device)

# 加载模型
ba_model.load_state_dict(torch.load('./save_model/20230228_new_model_40.bin'))


# 开始模型的预测
def model2predict():
    ba_model.eval()
    with torch.no_grad():
        for sentence, pos1, pos2, label, original_sequences, original_labels, entites in tqdm(test_iter):
            print(label)
            print(original_labels)
            print('original_sequences', len(original_sequences))
            print('original_labels', len(original_labels))
            # 将数据输入模型
            output = ba_model(sentence, pos1, pos2)
            # 实现模型的预测
            predict_ids = torch.argmax(output, dim=1).tolist()

            for i in range(len(original_sequences)):
                original_sequence = ''.join(original_sequences[i])
                original_label = id2relation[original_labels[i]]
                entity = entites[i]
                predict_label = id2relation[predict_ids[i]]
                print('原始句子: ', original_sequence)
                print('原始关系类别: ', original_label)
                print('实体列表',entity)
                print('模型预测的关系类别: ', predict_label)
                print('*'*80)
                break


if __name__ == '__main__':
    model2predict()