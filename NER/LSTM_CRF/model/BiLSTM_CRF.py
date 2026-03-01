import torch
import torch.nn as nn
from TorchCRF import CRF
from NER.LSTM_CRF.utils.data_loader import *

class NERLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM_CRF, self).__init__()
        self.name = "BiLSTM_CRF"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        #CRF
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size)
    def forward(self,x,mask):
        outputs = self.get_lstm2linear(x)
        outputs = outputs * mask.unsqueeze(-1)
        # 获取维特比解码后的预测标签
        outputs = self.crf.viterbi_decode(outputs,mask)
        return outputs
    def log_likelihood(self,x,tags,mask):
        outputs = self.get_lstm2linear(x)
        outputs = outputs * mask.unsqueeze(-1)
        return -self.crf(outputs,tags,mask)
    def get_lstm2linear(self,x):
        embedding = self.word_embeds(x)
        outputs,hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return outputs
if __name__ == '__main__':
    datas, word2id = build_data()
    conf = Config()
    ner_lstm_crf = NERLSTM_CRF(conf.embedding_dim, conf.hidden_dim,
                               conf.dropout, word2id, conf.tag2id)
    print(ner_lstm_crf)
    train_dataloader, dev_dataloader = get_data()
    for x, y, attention in train_dataloader:
        mask = attention.to(torch.bool)
        result = ner_lstm_crf.log_likelihood(x, y, mask)
        print(f'result--》{result}')
        break