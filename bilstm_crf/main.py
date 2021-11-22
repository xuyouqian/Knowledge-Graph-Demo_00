from config import Config
from utils import create_dataloader
from train import create_model, train

import torch

config = Config()
train_iter, dev_iter, test_iter = create_dataloader(config)
model = create_model(config)
epochs = config.epochs

optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.95, nesterov=True)
train(model, train_iter, dev_iter, optimizer, epochs)
