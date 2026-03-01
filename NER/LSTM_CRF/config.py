import os
import torch
import json
class Config(object):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        self.train_path = r"D:\AI_agent\Medical_Graph\Ner\LSTM_CRF\data\train.txt"
        self.vocab_path = r"D:\AI_agent\Medical_Graph\Ner\LSTM_CRF\vocab\vocab.txt"
        self.embedding_dim = 300
        self.epochs = 5
        self.batch_size = 8
        self.hidden_dim = 256
        self.lr = 2e-3
        self.dropout = 0.2
        self.model = "BiLSTM_CRF"
        self.tag2id = json.load(open(r"D:\AI_agent\Medical_Graph\Ner\LSTM_CRF\data\tag2id.json"))
if __name__ == "__main__":
    conf = Config()
    print(conf.train_path)
    print(conf.tag2id)