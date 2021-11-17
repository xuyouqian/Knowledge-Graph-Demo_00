import pandas as pd
import yaml
import re

from pyhanlp import HanLP
from tqdm import tqdm

'''
pip install pyyaml
'''


def translate(str):
    line = str.strip()  # 处理前进行相关的处理，包括转换成Unicode等
    pattern = re.compile('[^\u4e00-\u9fa50-9\.]')  # 中文的编码范围是：\u4e00到\u9fa5
    zh = "".join(pattern.split(line)).strip()
    return zh


def build_yaml_loader():
    loader = yaml.FullLoader
    # 这个是为了解决加载浮点数变为字符串的问题
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    return loader


if __name__ == '__main__':
    yaml_loader = build_yaml_loader()
    label_map = yaml.load(open('label_map.yaml').read(), Loader=yaml_loader)
    segment = HanLP.newSegment().enableCustomDictionaryForcing(True)
    train = []
    with open('toutiao_cat_data.txt', encoding='utf-8') as F:
        for line in tqdm(F):
            outs = []
            line = line.strip().split('_!')

            label = label_map[int(line[1][1:])]
            text = line[3][1:]
            tag = line[4][1:]

            text = translate(text)
            text = segment.seg(text)

            for word in text:
                nature = str(word.nature)
                if nature in ['ns', 'nr']:
                    outs.append(nature)
                elif word.word.replace('.', '').isdigit():
                    outs.append('num')
                else:
                    outs.append(word.word)
                    print(word.word)
            outs = ' '.join(outs)
            if not outs.replace(' ', ''):
                continue

            train.append([label, outs, tag])
            # line = translate(line)
    train = pd.DataFrame(train)
    train.columns = ['label', 'text', 'tag']
    train.to_csv('classfication.csv', index=False)
