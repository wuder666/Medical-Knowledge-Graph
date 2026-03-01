from Bilstm_Attention_RE.model.bilstm_attention import *
from utils.data_loader import *
from utils.process import *
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
def train(conf,vocab_size,pos_size,tag_size):
    train_dataloader,test_dataloader = get_loader_data()
    ba_model = BiLSTM_ATT(conf,vocab_size,pos_size,tag_size).to(conf.device)
    optimizer = optim.Adam(ba_model.parameters(),lr=conf.lr)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    train_loss = 0
    train_acc = 0
    total_iter_num = 0
    total_sample = 0
    for epoch in range(conf.epochs):
        for data, pos1, pos2, labels, _, _, _ in tqdm(train_dataloader):
            out = ba_model(data, pos1, pos2)
            loss = criterion(out,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iter_num += 1
            train_loss += loss.item()
            train_acc = train_acc+sum(torch.argmax(out,dim=1)==labels).item()
            total_sample = total_sample+labels.size()[0]
            if total_iter_num % 25 == 0:
                tmploss = train_loss / total_iter_num
                tmpacc = train_acc / total_sample
                end = time.time()
                print('轮次: %d, 损失:%.6f, 时间:%d, 准确率:%.3f' % (epoch + 1, tmploss, end - start, tmpacc))
        if epoch % 10 == 0:
            torch.save(ba_model.state_dict(), './save_model/20230228_new_model_%d.bin' % epoch)

if __name__ == '__main__':
    word2id, id2word = get_word_id(conf.train_path)
    vocab_size = len(word2id)
    print(vocab_size)
    pos_size = 143
    tag_size = len(relation2id)
    train(conf, vocab_size, pos_size, tag_size)


