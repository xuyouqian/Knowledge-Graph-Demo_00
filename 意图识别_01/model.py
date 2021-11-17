import torch.nn.functional as F  # pytorch 激活函数的类
import torch
from torch import nn, optim  # 构建模型和优化器


# 构建分类模型
class TextCNN(nn.Module):
    def __init__(self, parameter, vocab_size):
        super(TextCNN, self).__init__()
        filter_size = (3, 4, 5)
        hidden_size = parameter['hidden_size']
        embedding_dim = parameter['embedding_dim']
        output_size = parameter['output_size']
        dropout = parameter['dropout']
        self.name = 'textcnn'
        self.embedding = nn.Embedding(num_embeddings=vocab_size,  # 词向量的总长度
                                      embedding_dim=parameter['embedding_dim'])
        self.convs = nn.ModuleList([nn.Conv2d(1, hidden_size, (k, embedding_dim)) for k in filter_size])
        # ins:(batch:100,C:1,len(chars):40,embedding_dim:300)
        # covs:(3/4/5,300),kernel = 128
        # outs:(batch:100,C:128,40-3/4/5+1,300-300+1)->(100,128,38,1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * len(filter_size), output_size)

    def forward(self, x):
        # 添加通道的维度  1 通道
        x = self.embedding(x)
        x = x.unsqueeze(1)
        # [batch, channel, word_num, embedding_dim] = [N,C,H,W] -> (-1, 1, 20, 300)

        x = [F.sigmoid(conv(x)).squeeze(3) for conv in
             self.convs]  # len(filter_size) * (N, filter_num, H) -> 3 * (-1, 100, 18)
        # x[0] = batch size, out channel,height,weight=1  其中weight 维度已经丢掉了
        out_new = []
        for output in x:
            try:
                out_new.append(F.max_pool1d(output, output.shape[2].item()).squeeze(2))
            except:
                out_new.append(F.max_pool1d(output, output.shape[2]).squeeze(2))
        x = out_new
        x = torch.cat(x, 1)  # (N, filter_num * len(filter_size)) -> (163, 100 * 3)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 构建分类模型
class TextRNN(nn.Module):
    def __init__(self, parameter, vocab_size):
        super(TextRNN, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        output_size = parameter['output_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']

        self.name = 'rnn'
        self.embedding = nn.Embedding(num_embeddings=vocab_size,  # 词向量的总长度
                                      embedding_dim=parameter['embedding_dim'])
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class TextRCNN(nn.Module):
    def __init__(self, parameter, vocab_size):
        super(TextRCNN, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        output_size = parameter['output_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        self.name = 'rcnn'
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=parameter['embedding_dim'])
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.fc_for_concat = nn.Linear(hidden_size * 2 + embedding_dim, hidden_size * 2)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)

        out = self.fc_for_concat(torch.cat((x, out), 2))
        # 激活函数
        out = F.tanh(out)
        #         print(out.shape)
        out = out.permute(0, 2, 1)
        #         print(out.shape)
        try:
            out = F.max_pool1d(out, out.size(2).item())
        except:
            out = F.max_pool1d(out, out.size(2))
        #         print(out.shape)
        out = out.squeeze(-1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class TextRCNN(nn.Module):
    def __init__(self, parameter, vocab_size):
        super(TextRCNN, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        output_size = parameter['output_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        filter_size = (3, 4, 5)
        self.name = 'rcnn'
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=parameter['embedding_dim'])
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.fc_for_concat = nn.Linear(hidden_size * 2 + embedding_dim, hidden_size * 2)

        self.convs = nn.ModuleList([nn.Conv2d(1, hidden_size, (k, embedding_dim)) for k in filter_size])
        # ins:(batch:100,C:1,len(chars):40,embedding_dim:300)
        # covs:(3/4/5,300),kernel = 128
        # outs:(batch:100,C:128,40-3/4/5+1,300-300+1)->(100,128,38,1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * len(filter_size), output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)

        out = self.fc_for_concat(torch.cat((x, out), 2))
        # 激活函数
        out = F.tanh(out)
        #         print(out.shape)
        out = out.permute(0, 2, 1)
        #         print(out.shape)
        try:
            out = F.max_pool1d(out, out.size(2).item())
        except:
            out = F.max_pool1d(out, out.size(2))
        #         print(out.shape)
        out = out.squeeze(-1)
        out = self.dropout(out)
        out = self.fc(out)
        return out