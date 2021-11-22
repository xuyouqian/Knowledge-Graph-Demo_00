import torch
import numpy as np
from model import BilstmCrf


def create_model(config):
    return BilstmCrf(config).to(config.device)


def train(model, train_iter, valid_iter, optimizer, epochs):
    best_acc = 0

    for epoch in range(epochs):
        loss, best_acc = train_single_epoch(model, train_iter, valid_iter, optimizer, best_acc, epoch)
    print(epoch, ':', loss)


def train_single_epoch(model, train_iter, valid_iter, optimizer, best_acc, epoch):
    epoch_loss = 0
    epoch_count = 0

    for index, ((inputs, label), _) in enumerate(train_iter):
        model.train()
        inputs = inputs.permute(1, 0)
        label = label.permute(1, 0)
        loss = model.compute_loss(inputs, label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if index % 2 == 0:
            ##########################################################
            acc, matrix = eval(model, valid_iter, return_matrix=False)
            ##############################################################
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), model.name + '.h5')
                print('epoch:{},index:{},acc:{:.4f}'.format(epoch, index, acc))

    return epoch_loss / epoch_count, best_acc


def eval(model, valid_iter, return_matrix=False):
    model.eval()

    total_count = 0
    acc_count = 0
    class_count = model.fc.out_features
    matrix = np.zeros((class_count, 3))

    for (inputs, label), _ in valid_iter:
        inputs = inputs.permute(1, 0)
        label = label.permute(1, 0)
        out = model.decode(inputs)
        total_count += label.shape[0] * label.shape[1]
        out = torch.tensor(out).to(label.device)
        acc_count += torch.sum(out == label)
        if return_matrix:
            # 按类别统计准确情况
            for class_index in range(class_count):
                pred = out == class_index
                gold = label == class_index
                tp = pred[pred == gold]
                matrix[class_index, 0] += torch.sum(tp)
                matrix[class_index, 1] += torch.sum(pred)
                matrix[class_index, 2] += torch.sum(gold)
        else:
            matrix = None

    return acc_count / total_count, matrix


def test(model, test_iter, config):
    acc, matrix = eval(model, test_iter, True)
