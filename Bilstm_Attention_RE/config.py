import torch
class Config():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_path = r"D:\AI_agent\Medical_Graph\Bilstm_Attention_RE\data\train.txt"
        self.test_path = r"D:\AI_agent\Medical_Graph\Bilstm_Attention_RE\data\test.txt"
        self.relation2id = r"D:\AI_agent\Medical_Graph\Bilstm_Attention_RE\data\relation2id.txt"
        self.embedding_dim = 128
        self.hidden_dim = 200
        self.epochs = 50
        self.pos_dim = 32
        self.batch_size = 32
        self.max_len = 70
        self.lr = 1e-3
        