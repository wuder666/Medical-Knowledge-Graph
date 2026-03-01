import torch.nn as nn
import torch.optim as optim
from model.BiLSTM import *
from model.BiLSTM_CRF import *
from utils.data_loader import *
from tqdm import tqdm
conf = Config()
models = {"BiLSTM":NERLSTM,
          "BiLSTM_CRF":NERLSTM_CRF}
model = models[conf.model](conf.embedding_dim,conf.hidden_dim,conf.dropout,word2id,conf.tag2id)
model.load_state_dict(torch.load(r"D:\AI_agent\Medical_Graph\NER\LSTM_CRF\save_model\BiLSTM_CRF_best.pth"))
id2tag = {value:key for key,value in conf.tag2id.items()}
def model2test(sample):
    x = []
    for char in sample:
        if char not in word2id:
            char = "UNK"
        x.append(word2id[char])
    x_train = torch.tensor([x])
    mask = (x_train!=0).long()
    model.eval()
    with torch.no_grad():
        if model.name=="BiLSTM":
            outputs = model(x_train,mask)
            pred_ids = torch.argmax(outputs,dim=-1)[0]
            tags = [id2tag[i.item()] for i in pred_ids]
        else:
            pred_ids = model(x_train,mask)
            tags = [id2tag[i] for i in pred_ids[0]]
        chars = [i for i in sample]
        assert len(chars) == len(tags)
        result = extract_entities(chars,tags)
        return result
def extract_entities(char,tag):
    entities = []
    entity = []
    entity_type = None
    for char,tag in zip(char,tag):
        if tag.startswith("B-"):
            if entity:
                entities.append((entity_type,"".join(entity)))
                entity = []
            entity_type = tag.split("-")[1]
            entity.append(char)
        elif tag.startswith("I-") and entity:
            entity.append(char)
        else:
            if entity:
                entities.append((entity_type,"".join(entity)))
                entity = []
                entity_type = None
    if entity:
        entities.append((entity_type,"".join(entity)))
    return {entity:entity_type for entity_type,entity in entities}
if __name__ == '__main__':
    result = model2test(sample='小明的父亲患有冠心病及糖尿病，无手术外伤史及药物过敏史')
    print(result)