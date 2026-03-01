from Bilstm_Attention_RE.config import *
from itertools import chain
from collections import Counter
conf = Config()
relation2id = {}
with open(conf.relation2id,"r",encoding="utf-8") as fr:
    for line in fr.readlines():
        word,id = line.rstrip().split(" ")
        if word not in relation2id:
            relation2id[word] = int(id)
def sent_padding(words,word2id):
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id["UNKNOW"])
    if len(ids) >= conf.max_len:
        return ids[:conf.max_len]
    ids.extend([word2id["BLANK"]]*(conf.max_len-len(ids)))
    return ids
def pos(num):
    if num<-70:
        return 0
    elif num>=-70 and num<70:
        return num+70
    else:
        return 142
def position_padding(pos_ids):
    pos_ids = [pos(id) for id in pos_ids]
    if len(pos_ids) < conf.max_len:
        pos_ids.extend([142]*(conf.max_len-len(pos_ids)))
    pos_ids = pos_ids[:conf.max_len]
    return pos_ids
def get_txt_data(data_path):
    datas = []
    labels = []
    positionE1 = []
    positionE2 = []
    entities = []
    relation3id = {key:0 for key, value in relation2id.items()}
    with open(data_path,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            line = line.rstrip().split(" ",maxsplit=3)
            if line[2] not in relation3id:
                continue
            elif relation3id[line[2]] > 2000:
                continue
            else:
                entities.append([line[0],line[1]])
                sentence = []
                index1 = line[3].index(line[0])
                position1 = []
                index2 = line[3].index(line[1])
                position2 = []
                assert len(line) == 4
                for i,word in enumerate(line[3]):
                    sentence.append(word)
                    position1.append(i-index1)
                    position2.append(i-index2)
                datas.append(sentence)
                labels.append(relation2id[line[2]])
                positionE1.append(position1)
                positionE2.append(position2)
                relation3id[line[2]] += 1
    return datas,labels,positionE1,positionE2,entities
def get_word_id(data_path):
    datas,labels,positionE1,positionE2,entities = get_txt_data(data_path)
    datas = list(set(chain(*datas)))
    word2id = {value:key for key,value in enumerate(datas)}
    id2word = {key:value for key,value in enumerate(datas)}
    word2id["BLANK"] = len(word2id)
    word2id["UNKNOW"] = len(word2id)
    id2word[len(id2word)] = "BLANK"
    id2word[len(id2word)] = "UNKNOW"
    return word2id,id2word
if __name__ == '__main__':
    word2id, id2word = get_word_id(data_path=conf.train_path)
    print(f'word2id--》{word2id}')
    print(f'id2word--》{id2word}')
