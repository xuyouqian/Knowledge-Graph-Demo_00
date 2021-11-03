from gensim.models import VocabTransform, Word2Vec
import logging
import jieba
import re

from config import *

'''
基于已有语料，进行匹配对话或生成式对话

    数据预处理模块
'''

def load_seq_qa():
    '''
    加载现有语料
    '''
    q_list,a_list = [],[]
    with open(CORPUS_PATH,'r',encoding='utf-8') as f:
        for i,line in enumerate(f):
            line = re.sub(r'[0-9]*','',line.strip())
            line = jieba.lcut(line)
            if i % 2 == 0:
                q_list.append(line)
            else:
                a_list.append(line)
        return q_list,a_list

def build_vocab():
    """根据问句建立字典，键为词、值为序号"""
    q_list,_ = load_seq_qa()
    word_dict = set([j for i in q_list for j in i])
    word_dict = dict(zip(word_dict,range(3,len(word_dict)+3)))
    # word_dict['<pos>'] = 0
    # word_dict['<start>'] = 1
    # word_dict['<end>'] = 2
    return word_dict

def build_inverse():
    """根据问句构建倒排索引"""
    word_dict = build_vocab()
    q_list,_ = load_seq_qa()
    inverse_index_dict = {}
    for i in word_dict.keys():
        inverse_index_dict[i] = []
    for i, qw in enumerate(q_list):
            for w in qw:
                inverse_index_dict[w].append(i)
    return inverse_index_dict


