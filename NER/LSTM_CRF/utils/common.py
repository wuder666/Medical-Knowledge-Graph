from NER.LSTM_CRF.config import *
conf = Config()
"""
构造数据集，先对根据标注好的训练数据进行逐行遍历，跳过空行，避免遍历出错，
接着把每行的词写入x样本，每个词标注的数据写入y样本
如果这个词在词典中没有出现过就把这个词添加到词典中
接着如果遍历到句尾的标点符号，像。？这种，说明这一句话已经遍历完成了
就把整句话x样本跟标注的数据y样本添加到数据集中
接着通过遍历词典构建word:idx的词典
并把构建的数据集跟词典通过return返回
"""
def build_data():
    datas = []
    sample_x = []
    sample_y = []
    vocab_list = ["PAD","UNK"]
    for line in open(conf.train_path,"r",encoding="utf-8"):
        line_data = line.rstrip().split("\t")
        if not line_data:
            continue
        char = line_data[0]
        if not char:
            continue
        sample_x.append(char)
        label = line_data[-1]
        sample_y.append(label)
        if char not in vocab_list:
            vocab_list.append(char)
        if char in ['。', '?', '!', '！', '？']:
            datas.append([sample_x,sample_y])
            sample_x = []
            sample_y = []
    word2id = {wd:idx for idx,wd in enumerate(vocab_list)}
    write_file(vocab_list,conf.vocab_path)
    return datas,word2id
def write_file(vocab,vocab_path):
    with open(vocab_path,"w",encoding="utf-8") as f:
        f.write("\n".join(vocab))
if __name__ == "__main__":
    datas,word2id = build_data()
    print(len(datas))
    print(datas[:1])
    print(word2id)
    print(len(word2id))



