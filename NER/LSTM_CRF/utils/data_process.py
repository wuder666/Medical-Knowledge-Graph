import json
import os
os.chdir("..")
cur = os.getcwd()
print("当前数据处理默认工作目录:",cur)
"""
把训练数据中文件中的每个文字进行实体标注
B是实体开头，I是实体内部，O是非实体
"""
class TransferData():
    def __init__(self):
        self.label_dict = json.load(open(os.path.join(cur,"data/labels.json"),'r', encoding='utf-8'))
        self.seq_tag_dict = json.load(open(os.path.join(cur,"data/tag2id.json"),'r', encoding='utf-8'))
        self.origin_path = os.path.join(cur,"data_origin")
        self.train_filepath = os.path.join(cur,"data/train.txt")
    def transfer(self):
        with open(self.train_filepath,"w",encoding="utf-8") as fr:
            for root,dirs,files in os.walk(self.origin_path):
                for file in files:
                    filepath = os.path.join(root,file)
                    if "original" not in filepath:
                        continue
                    label_filepath = filepath.replace(".txtoriginal","")
                    # print(filepath, '\n',label_filepath)
                    res_dict = self.read_label_text(label_filepath)
                    with open(filepath,"r",encoding="utf-8") as f:
                        content = f.read().strip()
                        for idx,word in enumerate(content):
                            idx_label = res_dict.get(idx,"O")
                            fr.write(word + "\t" + idx_label + "\n")
    def read_label_text(self,label_filepath):
        res_dict = {}
        for line in open(label_filepath,"r",encoding="utf-8"):
            res = line.strip().split("\t")
            start = int(res[1])
            end = int(res[2])
            data = res[3]
            data_label = self.label_dict.get(data)
            for i in range(start,end+1):
                if i == start:
                    res_dict[i] = "B-"+data_label
                else:
                    res_dict[i] = "I-"+data_label
        return res_dict
if __name__ == "__main__":
    handler = TransferData()
    handler.transfer()