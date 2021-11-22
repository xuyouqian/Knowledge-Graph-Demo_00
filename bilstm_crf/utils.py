import json

from torchtext.legacy.data import Field
from torchtext.legacy import data
from tqdm import tqdm


# 读取文件
def read_and_process(path):
    list_data = []
    with open(path, encoding='utf-8') as F:
        for line in F:
            line = json.loads(line)
            text = line['text']
            if not line.get('label'):
                text = list(text)
                label = len(text) * ['O']
            else:
                label = line['label']
                text, label = transform_sample(text, label)
            list_data.append((text, label))
    return list_data


class MyDataset(data.Dataset):
    def __init__(self, datatuple, text_field, label_field, test=False):  # datatuple指的是元组('this moive is great',1)
        fields = [("text", text_field), ("label", label_field)]
        lists = []
        if test:
            # 如果为测试集，则不加载label
            for content, label in tqdm(datatuple):
                lists.append(data.Example.fromlist([content, None], fields))
        else:
            for content, label in tqdm(datatuple):
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                lists.append(data.Example.fromlist([content, label], fields))
        # 之前是一些预处理操作，此处调用super初始化父类，构造自定义的Dataset类
        super().__init__(lists, fields)


def transform_sample(text, label):
    """
    text 字符串 :'突袭黑暗雅典娜》中Riddick发现之前抓住他的赏金猎人Johns，'
    label : 字典  {'game': {'突袭黑暗雅典娜》': [[0, 7]]}, 'name': {'Riddick': [[9, 15]], 'Johns': [[28, 32]]}}
    """
    text = list(text)
    count = len(text)
    processed_label = ['O'] * count
    for key, value in label.items():
        label_indexes = value.values()
        # start_idx: 实体开始索引
        # end_idx: 实体结束索引
        for label_index in label_indexes:
            for start_idx, end_idx in label_index:

                if start_idx == end_idx:
                    processed_label[start_idx] = 'S-' + key

                elif start_idx + 1 == end_idx:
                    processed_label[start_idx:end_idx + 1] = ['B-' + key, 'E-' + key]

                elif end_idx - start_idx > 1:
                    new_labels = ['B-' + key] + ['I-' + key] * (end_idx - start_idx - 1) + ['E-' + key]
                    processed_label[start_idx:end_idx + 1] = new_labels

    return text, processed_label


def create_dataset(data_list, config, is_train=True):
    if is_train:
        SRC = Field(tokenize=lambda x: x, fix_length=config.fix_length)
        LABEL = Field(tokenize=lambda x: x, fix_length=config.fix_length)  # 针对文本分类的类别标签
        config.SRC = SRC
        config.LABEL = LABEL
    else:
        SRC = config.SRC
        LABEL = config.LABEL

    return MyDataset(data_list, SRC, LABEL), SRC, LABEL


def built_iter(dataset, config):
    return data.BucketIterator(dataset=dataset, batch_size=config.batch_size,
                               shuffle=True, sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False,
                               device=config.device)


def create_dataloader(config):
    train_data_list = read_and_process(config.train_path)
    dev_data_list = read_and_process(config.dev_path)
    test_data_list = read_and_process(config.test_path)

    SRC = Field(tokenize=lambda x: x, fix_length=config.fix_length)
    LABEL = Field(tokenize=lambda x: x, fix_length=config.fix_length)

    train_dataset = MyDataset(train_data_list, SRC, LABEL)
    dev_dataset = MyDataset(dev_data_list, SRC, LABEL)
    test_dataset = MyDataset(test_data_list, SRC, LABEL)

    SRC.build_vocab(train_dataset)
    LABEL.build_vocab(train_dataset)

    LABEL.vocab.stoi.pop('<unk>')
    LABEL.vocab.stoi[LABEL.vocab.itos[-1]] = 0
    LABEL.vocab.itos = sorted(LABEL.vocab.stoi, key=lambda x: LABEL.vocab.stoi[x])

    config.SRC = SRC
    config.LABEL = LABEL

    train_iter, dev_iter, test_iter = map(built_iter,
                                          [train_dataset, dev_dataset, test_dataset], [config] * 3)

    return train_iter, dev_iter, test_iter


if __name__ == '__main__':
    from config import Config

    data_list = read_and_process('data/train.json')
    config = Config()
    dataset, _, _ = create_dataset(data_list, config, is_train=True)


    from config import Config

    config = Config()

    train_iter, dev_iter, test_iter = create_dataloader(config)
    for (a, b), _ in train_iter:
        print(a.shape)
        print(b.shape)
        break
