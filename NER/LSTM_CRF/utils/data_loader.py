from NER.LSTM_CRF.utils.common import *
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import torch
"""
构造数据迭代器
"""
datas,word2id = build_data()
class NerDataset(Dataset):
    def __init__(self,datas):
        super().__init__()
        self.datas = datas
    def __len__(self):
        return len(self.datas)
    def __getitem__(self,item):
        x = self.datas[item][0]
        y = self.datas[item][1]
        return x, y
def collate_fn(batch):
    train_tensor = [torch.tensor([word2id[char] for char in line[0]]) for line in batch]
    label_tensor = [torch.tensor([conf.tag2id[label] for label in line[1]]) for line in batch]
    train_tensor_pad = pad_sequence(train_tensor,batch_first=True,padding_value=0)
    label_tensor_pad = pad_sequence(label_tensor,batch_first=True,padding_value=0)
    attention_mask = (train_tensor_pad!=0).long()
    return train_tensor_pad,label_tensor_pad,attention_mask
def get_data():
    train_data = NerDataset(datas[:6200])
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
    )
    dev_data = NerDataset(datas[6200:])
    dev_dataloader = DataLoader(
        dataset=dev_data,
        batch_size=conf.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return train_dataloader,dev_dataloader
if __name__ == "__main__":
    train_dataloader,dev_dataloader = get_data()
    for train_tensor_pad,label_tensor_pad,attention_mask in train_dataloader:
        # 1. 打印第一个批次的整体形状（确认维度）
        print("【1】第一个批次张量形状：")
        print(f"字符ID张量形状: {train_tensor_pad.shape}")
        print(f"标签ID张量形状: {label_tensor_pad.shape}")
        print(f"注意力掩码形状: {attention_mask.shape}")

        # 2. 打印第一个批次的第一条样本（最常用：看单条数据）
        print("\n【2】第一个批次的第一条样本：")
        print(f"字符ID张量: {train_tensor_pad[1]}")  # 取批次内第0个样本（第一条）
        print(f"标签ID张量: {label_tensor_pad[1]}")
        print(f"注意力掩码: {attention_mask[1]}")
        break  # 只打印第一个批次，终止循环