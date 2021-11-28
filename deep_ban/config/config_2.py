import torch

class Config:
    def __init__(self):

        self.fix_length = 20
        self.batch_size = 128
        self.embedding_dim = 300

        self.hid_dim = 128
        self.n_layers = 2
        self.dropout = 0.1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 300
        self.lr = 0.1

        self.output_size = 18
        self.vector_path = '../input/chinese-wiki-embedding/chinese_wiki_embeding20000.300d.txt'