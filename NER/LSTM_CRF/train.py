import torch
import torch.nn as nn
import torch.optim as optim
from NER.LSTM_CRF.model.BiLSTM import *
from NER.LSTM_CRF.model.BiLSTM_CRF import *
from NER.LSTM_CRF.utils.data_loader import *
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report
from config import *
import time
conf = Config()

def model2train():
    train_dataloader,dev_dataloader=get_data()
    models = {"BiLSTM":NERLSTM,
              "BiLSTM_CRF":NERLSTM_CRF}
    model = models[conf.model](conf.embedding_dim,conf.hidden_dim,conf.dropout,word2id,conf.tag2id)
    model.to(conf.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=conf.lr)
    start_time = time.time()
    if conf.model == "BiLSTM":
        f1_sum = -111
        for epoch in range(conf.epochs):
            model.train()
            for index,(char_tensor,label_tensor,mask) in enumerate(tqdm(train_dataloader,desc="BiLSTM训练")):
                x = char_tensor.to(conf.device)
                y = label_tensor.to(conf.device)
                mask = mask.to(conf.device)
                pred = model(x,mask)
                pred = pred.view(-1,len(conf.tag2id))
                loss = criterion(pred,y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if index % 200 == 0:
                    print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))
            precision,recall,f1,report = model2dev(dev_dataloader,model,criterion)
            if f1 > f1_sum:
                f1_sum = f1
                torch.save(model.state_dict(),r"D:\AI_agent\Medical_Graph\NER\LSTM_CRF\save_model\BiLSTM_best.pth")
                print(report)
        end_time = time.time()
        print(f"训练总耗时:{end_time-start_time}")
    elif conf.model == "BiLSTM_CRF":
        f1_sum = -111
        for epoch in range(conf.epochs):
            model.train()
            for index,(char_tensor,label_tensor,mask) in enumerate(tqdm(train_dataloader,desc="BiLSTM_CRF训练")):
                x = char_tensor.to(conf.device)
                y = label_tensor.to(conf.device)
                mask = mask.to(conf.device)
                mask = mask.bool()
                loss = model.log_likelihood(x,y,mask).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if index % 200 == 0:
                    print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))
            precision, recall, f1, report = model2dev(dev_dataloader, model, criterion)
            if f1 > f1_sum:
                f1_sum = f1
                torch.save(model.state_dict(), r"D:\AI_agent\Medical_Graph\NER\LSTM_CRF\save_model\BiLSTM_CRF_best.pth")
                print(report)
        end_time = time.time()
        print(f"训练总耗时:{end_time - start_time}")
def model2dev(dev,model,criterion=None):
    loss_sum = 0
    preds,golds = [],[]
    model.eval()
    for index,(char_tensor,label_tensor,mask) in enumerate(tqdm(dev,desc="验证集验证")):
        x = char_tensor.to(conf.device)
        y = label_tensor.to(conf.device)
        mask = mask.to(conf.device)
        predict = []
        if model.name == "BiLSTM":
            pred = model(x,mask)
            predict = torch.argmax(pred,dim=-1).tolist()
            pred = pred.view(-1,len(conf.tag2id))
            loss = criterion(pred,y.view(-1))
            loss_sum += loss.item()
        elif model.name == "BiLSTM_CRF":
            mask = mask.to(torch.bool)
            predict = model(x,mask)
            loss = model.log_likelihood(x,y,mask)
            loss_sum += loss.mean().item()
        leng = []
        for i in y.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j)
            leng.append(tmp)
        for index,label in enumerate(predict):
            preds.extend(label[:len(leng[index])])
        for index,label in enumerate(y.tolist()):
            golds.extend(label[:len(leng[index])])
    loss_sum /= (len(dev) * 64)
    precision = precision_score(preds,golds,average='macro')
    recall = recall_score(preds,golds,average='macro')
    f1 = f1_score(preds,golds,average='macro')
    report = classification_report(golds,preds)
    return precision,recall,f1,report
if __name__ == "__main__":
    model2train()



