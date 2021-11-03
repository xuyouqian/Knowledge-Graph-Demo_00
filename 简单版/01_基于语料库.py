import xml.etree.ElementTree as et

CORPUS_PATH = 'data/corpus/conversation_test.txt'

import jieba

# 构建标准问题和标准答案
def load_seq_qa():
    q_list, a_list = [], []
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for ind, i in enumerate(f):
            i = jieba.lcut(i.strip())
            if ind % 2 == 0:
                q_list.append(i)
            else:
                a_list.append(i)
    return q_list, a_list


# 构建单词表
def build_vocab():
    q, _ = load_seq_qa()
    word_dict = set([j for i in q for j in i])
    word_dict = dict(zip(word_dict, range(len(word_dict))))
    return word_dict



# 构建单词表示
def build_word_embeding():
    q, _ = load_seq_qa()
    word_dict = build_vocab()
    word_embeding = {}
    for w in word_dict.keys():
        word_embeding[w] = []
    for ind, qs in enumerate(q):
        for w in qs:
            word_embeding[w].append(ind)
    return word_embeding



from collections import Counter
import numpy as np
import jieba


class corpusSearch():
    def __init__(self):
        self.q, self.a = load_seq_qa()
        self.word_embeding = build_word_embeding()
        self.threshold = 0.7

    def cosine_sim(self, a, b):
        '''
        比较a 和 b 的相似度
        '''
        a_count = Counter(a)
        b_count = Counter(b)
        a_vec = []
        b_vec = []
        all_word = set(a + b)
        # 这边类似构建了一个句子的embeding
        for i in all_word:
            a_vec.append(a_count.get(i, 0))
            b_vec.append(b_count.get(i, 0))
        # 计算余弦相似
        a_vec = np.array(a_vec)
        b_vec = np.array(b_vec)
        cos = sum(a_vec * b_vec) / (sum(a_vec * a_vec) ** 0.5) / (sum(b_vec * b_vec) ** 0.5)
        return round(cos, 4)

    def search_answer(self, question):
        search_list = []
        q_words = jieba.lcut(question)
        for q_word in q_words:
            index = self.word_embeding.get(q_word, list())
            search_list += index
        if len(search_list) == 0:
            return None
        # 挑选前3个
        count_list = Counter(search_list)
        count_list = count_list.most_common(3)
        res_list = []
        for i, _ in count_list:
            q = self.q[i]
            sim = self.cosine_sim(q, q_words)
            if sim > self.threshold:
                res_list.append([sim, self.a[i]])
        res_list.sort()
        if len(res_list) == 0:
            return None
        else:
            return ''.join(res_list[-1][1])


test = corpusSearch()
while 1:
    question = input('请输入->')
    if question == ':':
        break
    print(test.search_answer(question))
