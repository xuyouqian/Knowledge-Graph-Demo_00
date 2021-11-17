import os
import shutil
import torch
import pickle as pk

from torch import nn
from model import TextCNN
from parameter import parameter
from utils import create_data_loader
from train import create_model, train
from torch.utils.tensorboard import SummaryWriter

# 记录日志
shutil.rmtree('textcnn') if os.path.exists('textcnn') else 1
writer = SummaryWriter('./textcnn', comment='textcnn')

train_iter, valid_iter, SRC = create_data_loader(parameter['data_path'], parameter['batch_size'],
                                                 parameter['embedding_path'], parameter['cuda'],
                                                 parameter['train_data_path'], parameter['valid_data_path'])

vocab_size = len(SRC.vocab)
model = create_model(TextCNN, parameter, SRC)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.95, nesterov=True)
criterion = nn.CrossEntropyLoss()
# train(model, train_iter, valid_iter, optimizer, writer, criterion, parameter['epoch'])

for i in train_iter:
    print(i)