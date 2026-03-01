import torch
import torch.nn as nn
import torch.nn.functional as F
from Bilstm_Attention_RE.config import *
conf = Config()
class BiLSTM_ATT(nn.Module):
    def __init__(self,conf,vocab_size,pos_size,tag_size):
        super(BiLSTM_ATT,self).__init__()
        self.batch = conf.batch_size
        self.device = conf.device
        self.embedding_dim = conf.embedding_dim
        self.hidden_dim = conf.hidden_dim
        self.pos_dim = conf.pos_dim
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.tag_size = tag_size
        self.word_embedding = nn.Embedding(self.vocab_size,
                                           self.embedding_dim)
        self.pos1_embedding = nn.Embedding(self.pos_size,
                                           self.pos_dim)
        self.pos2_embedding = nn.Embedding(self.pos_size,
                                           self.pos_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.pos_dim*2,
                            hidden_size=self.hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim,
                                self.tag_size)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.dropout_lstm = nn.Dropout(p=0.2)
        self.dropout_att = nn.Dropout(p=0.2)
        self.att_weight = nn.Parameter(torch.randn(self.batch,
                                                    1,
                                                    self.hidden_dim).to(self.device))
    def init_hidden_lstm(self):
        return (torch.randn(2,self.batch,self.hidden_dim//2).to(self.device),
                torch.randn(2,self.batch,self.hidden_dim//2).to(self.device))

    def attention(self, H):
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight, M), dim=-1)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)
    def forward(self,sentence,pos1,pos2):
        init_hidden = self.init_hidden_lstm()
        sentence_embedding = self.word_embedding(sentence)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        embedding_cat = torch.cat((sentence_embedding,
                                  pos1_embedding,
                                   pos2_embedding),2)
        embeds = self.dropout_embed(embedding_cat)
        embeds = torch.transpose(embeds,0,1)
        lstm_out,lstm_hidden = self.lstm(embeds,init_hidden)
        lstm_out = lstm_out.permute(1,2,0)
        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out))
        att_out = self.dropout_att(att_out).squeeze()
        result = self.linear(att_out)
        return result
if __name__ == '__main__':
    from Bilstm_Attention_RE.utils.process import *
    from Bilstm_Attention_RE.utils.data_loader import *
    conf = Config()
    word2id, id2word = get_word_id(data_path=conf.train_path)
    vocab_size = len(word2id)
    pos_size = 143
    print(f'relation2id--》{relation2id}')
    tag_size = len(relation2id)
    bl = BiLSTM_ATT(conf, vocab_size, pos_size, tag_size)
    bl.to(conf.device)
    train_dataloader, test_dataloader = get_loader_data()
    for setence, pos1, pos2, labels, _, _, _ in train_dataloader:
        print(bl(setence, pos1, pos2).shape)