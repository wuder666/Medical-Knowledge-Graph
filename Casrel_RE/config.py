# 导入必备的工具包
import torch
# 导入Vocabulary，目的：用于构建, 存储和使用 `str` 到 `int` 的一一映射
from fastNLP import Vocabulary
from transformers import BertTokenizer, AdamW
import json


# 构建配置文件Config类
class Config(object):
    def __init__(self):
        # 设置是否使用GPU来进行模型训练
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'mps'
        self.bert_path = r"D:\AI_agent\bert-base-chinese"
        self.num_rel = 18  # 关系的种类数
        self.batch_size = 4
        self.train_data_path = r"D:\AI_agent\Medical_Graph\Casrel_RE\data\train.json"
        self.dev_data_path = r"D:\AI_agent\Medical_Graph\Casrel_RE\data\dev.json"
        self.test_data_path = r"D:\AI_agent\Medical_Graph\Casrel_RE\data\test.json"
        self.rel_dict_path = r"D:\AI_agent\Medical_Graph\Casrel_RE\data\relation.json"
        id2rel = json.load(open(self.rel_dict_path, encoding='utf8'))
        # print(f'id2rel--》{id2rel}')
        # self.rel2id = {value: int(key) for key, value in id2rel.items()}
        self.rel_vocab = Vocabulary(padding=None, unknown=None)
        # vocab更新自己的字典，输入为list列表
        self.rel_vocab.add_word_lst(list(id2rel.values()))
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.learning_rate = 1e-5
        self.bert_dim = 768
        self.epochs = 5

if __name__ == '__main__':
    conf = Config()
    print(f'conf.rel_vocab--->{len(conf.rel_vocab)}')
    print(conf.rel_vocab.word2idx["所属专辑"])
    print(conf.rel_vocab.to_index("所属专辑"))


