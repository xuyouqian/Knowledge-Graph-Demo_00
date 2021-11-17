import torch

parameter = {
    'min_count_word': 1,
    'output_size': 15,
    'epoch': 20,
    'batch_size': 10,
    'embedding_dim': 300,
    'hidden_size': 128,
    'num_layers': 2,  # 堆叠LSTM的层数，默认值为1
    'dropout': 0.5,
    'cuda': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'lr': 0.001,
    'num_unknow': 0,
    'max_len': 20,
    'data_path': 'data/classfication.csv',
    'embedding_path': 'data/chinese_wiki_embeding8000.300d.txt',
    'src_path': 'data/src.pkl',
    'train_data_path': 'data/train_data.pkl',
    'valid_data_path': 'data/valid_data.pkl'
}
