"""

pip install pytorch-crf==0.7.2
"""

import torch.nn.functional as F  # pytorch 激活函数的类
from torch import nn, optim  # 构建模型和优化器
from torchcrf import CRF


# 构建基于bilstm+crf实现ner
class BilstmCrf(nn.Module):
    def __init__(self, config):
        super(BilstmCrf, self).__init__()
        self.name = 'bilstm_crf'
        SRC = config.SRC
        LABEL = config.LABEL
        word_size = len(SRC.vocab)
        embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(word_size, embedding_dim)

        hidden_size = config.hid_dim
        num_layers = config.n_layers
        dropout = config.dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)

        output_size = len(LABEL.vocab)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.crf = CRF(output_size, batch_first=True)

    def forward(self, x):
        out = self.embedding(x)
        out, (h, c) = self.lstm(out)
        out = self.fc(out)
        return out

    def compute_loss(self, x, y):
        out = self.forward(x)
        loss = -self.crf(out, y)
        return loss

    def decode(self, x):
        out = self.forward(x)
        predicted_index = self.crf.decode(out)
        return predicted_index

# loss = -model.crf(out,targets.long().to(parameter['device']))
