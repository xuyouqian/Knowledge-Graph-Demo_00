import os
import torch

class Config:
    def __init__(self):
        self.label_map = {'O': 0,
                          'B-name': 1,
                          'I-name': 2,
                          'E-name': 3,
                          'B-company': 4,
                          'I-company': 5,
                          'E-company': 6,
                          'B-game': 7,
                          'I-game': 8,
                          'E-game': 9,
                          'B-organization': 10,
                          'I-organization': 11,
                          'E-organization': 12,
                          'B-movie': 13,
                          'I-movie': 14,
                          'E-movie': 15,
                          'B-address': 16,
                          'E-address': 17,
                          'B-position': 18,
                          'I-position': 19,
                          'E-position': 20,
                          'B-government': 21,
                          'I-government': 22,
                          'E-government': 23,
                          'B-scene': 24,
                          'I-scene': 25,
                          'E-scene': 26,
                          'I-address': 27,
                          'B-book': 28,
                          'I-book': 29,
                          'E-book': 30,
                          'S-company': 31,
                          'S-address': 32,
                          'S-name': 33,
                          'S-position': 34}
        self.SRC = None
        self.LABEL = None

        self.train_path = 'data/train.json'
        self.dev_path = 'data/dev.json'
        self.test_path = 'data/test.json'

        abspath = os.path.abspath('')
        # 把相对路径改成绝对路径
        self.test_path, self.dev_path, self.test_path = map(lambda x: os.path.join(abspath, x),
                                                            [self.test_path, self.dev_path, self.test_path])

        self.fix_length = 50
        self.batch_size = 64
        self.embedding_dim = 300

        self.hid_dim = 256
        self.n_layers = 3
        self.dropout = 0.5

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 5


class Config:
    def __init__(self):
        self.SRC = None
        self.LABEL = None

        self.train_path = 'data/train.json'
        self.dev_path = 'data/dev.json'
        self.test_path = 'data/test.json'

        self.fix_length = 50
        self.batch_size = 100
        self.embedding_dim = 768

        self.hid_dim = 300
        self.n_layers = 2
        self.dropout = 0.1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 30
        self.lr = 0.00005
        self.momentum = 0.95