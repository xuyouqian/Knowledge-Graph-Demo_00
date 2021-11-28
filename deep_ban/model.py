import torch
from torch import nn
from torchcrf import CRF

# 构建分类模型
class TextRNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(TextRNN, self).__init__()
        embedding_dim = config.embedding_dim
        hidden_size = config.hid_dim
        output_size = config.output_size
        num_layers = config.n_layers
        dropout = config.dropout

        self.name = 'rnn'
        self.embedding = nn.Embedding(num_embeddings=vocab_size,  # 词向量的总长度
                                      embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


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


def create_model(config, SRC):
    vocab_size = len(SRC.vocab)
    model = TextRNN(config, vocab_size)
    vocab_vectors = SRC.vocab.vectors.numpy()  # 准备好预训练词向量
    model.embedding.weight.data.copy_(torch.from_numpy(vocab_vectors))

    model.embedding.weight.requires_grad = True
    return model.to(config.device)
