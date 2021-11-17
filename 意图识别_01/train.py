import torch
import numpy as np


def create_model(model_class, parameter, SRC):
    vocab_size = len(SRC.vocab)
    model = model_class(parameter, vocab_size)
    vocab_vectors = SRC.vocab.vectors.numpy()  # 准备好预训练词向量
    model.embedding.weight.data.copy_(torch.from_numpy(vocab_vectors))

    model.embedding.weight.requires_grad = True
    return model.to(parameter['cuda'])


def train(model, train_iter, valid_iter, optimizer,  criterion, epochs):
    (inputs, label), _ = next(iter(train_iter))
    # seqs, label, keys, epoch = next(train_iter)

    best_acc = 0

    for epoch in range(epochs):
        loss, best_acc = train_single_epoch(model, train_iter, valid_iter, optimizer, criterion, best_acc, epoch)
    print(epoch, ':', loss)


def train_single_epoch(model, train_iter, valid_iter, optimizer, criterion, best_acc, epoch):
    model.train()
    epoch_loss = 0
    epoch_count = 0

    for index, ((inputs, label), _) in enumerate(train_iter):
        inputs = inputs.permute(1, 0)
        out = model(inputs)
        loss = criterion(out, label)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_count += inputs.shape[0]
        if index % 500 == 0:
            acc, matrix = eval(model, valid_iter)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), model.name + '.h5')
                print('epoch:{},index:{},acc:{:.4f}'.format(epoch, index, acc))

    return epoch_loss / epoch_count, best_acc


def eval(model, valid_iter):
    model.eval()

    total_count = 0
    acc_count = 0
    class_count = model.fc.out_features
    matrix = np.zeros((class_count, 6))

    for (inputs, label), _ in valid_iter:
        inputs = inputs.permute(1, 0)
        out = model(inputs)
        _, out = torch.max(out, dim=-1)
        total_count += out.shape[0]
        acc_count += torch.sum(out == label)
        # 按类别统计准确情况
        # for class_index in range(class_count):
        #     pred = out == class_index
        #     gold = label == class_index
        #     tp = pred == gold
        #     matrix[class_index, 0] += sum(tp)
        #     matrix[class_index, 1] += sum(pred)
        #     matrix[class_index, 2] += sum(gold)

    return acc_count / total_count, matrix
