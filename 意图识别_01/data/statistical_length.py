"""
统计训练数据问文本长度
"""
import pandas as pd

df = pd.read_csv('classfication.csv')
L = []
print(df)
for text in df['text']:
    L.append(len(text.split(' ')))

L.sort()
print(L[-1])
print(L[int(len(L) * 0.9)])
print(L[int(len(L) * 0.8)])
print(L[int(len(L) * 0.95)])
print(L[int(len(L) * 0.98)])
