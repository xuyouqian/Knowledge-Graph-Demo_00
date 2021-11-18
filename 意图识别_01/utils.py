import os
import csv
import torch
import pickle as pk

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 新版导包
from torchtext.legacy.data import Field
from torchtext.legacy import data
from torchtext.vocab import Vectors


# 旧版导包
# from torchtext.data import Field
# from torchtext import data


# 读取文件
def read_and_process(path):
    list_data = []
    with open(path, encoding='utf-8') as F:
        f_csv = csv.reader((F))
        for line in f_csv:
            label, text, _ = line
            list_data.append((label, text))
    return list_data


def tokenizer(text):
    """分词"""
    return text.split(' ')


# 定义Dataset
class MyDataset(data.Dataset):

    def __init__(self, datatuple, text_field, label_field, test=False):
        # datatuple指的是元组('this moive is great',1)
        fields = [("text", text_field), ("label", label_field)]
        lists = []
        if test:
            # 如果为测试集，则不加载label
            for label, content in tqdm(datatuple):
                lists.append(data.Example.fromlist([content, None], fields))
        else:
            for label, content in tqdm(datatuple):
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                lists.append(data.Example.fromlist([content, label], fields))
        # 之前是一些预处理操作，此处调用super初始化父类，构造自定义的Dataset类
        super().__init__(lists, fields)


def create_data_loader(path, batch_size, vector_path, device):
    list_data = read_and_process(path)
    train_data, valid_data = train_test_split(list_data, test_size=0.3, shuffle=True)

    SRC = Field(tokenize=tokenizer, fix_length=20)
    LABEL = Field(sequential=False, use_vocab=False)  # 针对文本分类的类别标签
    train_dataset = MyDataset(train_data, SRC, LABEL)
    valid_dataset = MyDataset(valid_data, SRC, LABEL)

    vectors = Vectors(name=vector_path, unk_init=torch.Tensor.normal_)
    SRC.build_vocab(train_dataset, vectors=vectors)

    train_iter = data.BucketIterator(dataset=train_dataset, batch_size=batch_size,
                                     shuffle=True, sort_key=lambda x: len(x.text), sort_within_batch=False,
                                     repeat=False, device=device)
    dev_iter = data.BucketIterator(dataset=valid_dataset, batch_size=batch_size,
                                   shuffle=True, sort_key=lambda x: len(x.text), sort_within_batch=False,
                                   repeat=False, device=device)

    return train_iter, dev_iter, SRC,train_dataset


if __name__ == '__main__':
    train_iter, dev_iter, SRC ,train_dataset= create_data_loader('data/classfication.csv', 30,
                                                   'data/chinese_wiki_embeding8000.300d.txt', 'cpu')

    for (inputs, label), _ in train_iter:

        pass

    # for i in train_dataset:
    #     print(i.text)
    #     break
