from Bilstm_Attention_RE.utils.process import *
from torch.utils.data import DataLoader,Dataset
import os
import torch
class MyDataset(Dataset):
    def __init__(self,data_path):
        self.data = get_txt_data(data_path)
    def __len__(self):
        return len(self.data[0])
    def __getitem__(self,index):
        sequence = self.data[0][index]
        label = int(self.data[1][index])
        positionE1 = self.data[2][index]
        positionE2 = self.data[3][index]
        entities = self.data[4][index]
        return sequence,label,positionE1,positionE2,entities
word2id,id2word = get_word_id(conf.train_path)
def collate_fn(datas):
    sequences = [data[0] for data in datas]
    labels = [data[1] for data in datas]
    positionE1 = [data[2] for data in datas]
    positionE2 = [data[3] for data in datas]
    entities = [data[4] for data in datas]
    sequence_id = []
    for words in sequences:
        sequence_id.append(sent_padding(words,word2id))
    positionE1_ids = []
    positionE2_ids = []
    for position1 in positionE1:
        positionE1_ids.append(position_padding(position1))
    for position2 in positionE2:
        positionE2_ids.append(position_padding(position2))
    data_tensor = torch.tensor(sequence_id,
                               dtype=torch.long,
                               device=conf.device)
    positionE1_tensor = torch.tensor(positionE1_ids,
                                     dtype=torch.long,
                                     device=conf.device)
    positionE2_tensor = torch.tensor(positionE2_ids,
                                     dtype=torch.long,
                                     device=conf.device)
    labels_tensor = torch.tensor(labels,
                                 dtype=torch.long,
                                 device=conf.device)
    return data_tensor,positionE1_tensor,positionE2_tensor,labels_tensor,sequences,labels,entities
def get_loader_data():
    train_data = MyDataset(conf.train_path)
    train_loader = DataLoader(train_data,
                              batch_size=conf.batch_size,
                              shuffle=False,
                              collate_fn=collate_fn,
                              drop_last=True)
    test_data = MyDataset(conf.test_path)
    test_loader = DataLoader(test_data,
                             batch_size=conf.batch_size,
                             shuffle=False,
                             collate_fn=collate_fn,
                             drop_last=True)
    return train_loader,test_loader

if __name__ == '__main__':
    train_dataloader, test_dataloader = get_loader_data()
    for input, pos1, pos2, labels, _, _, _ in train_dataloader:
        print(f'input--》{input}')
        print(f'pos1--》{pos1.shape}')
        print(f'pos2--》{pos2.shape}')
        print(f'labels--》{labels.shape}')
        break