import xml.etree.ElementTree as et

template = et.parse('data/robot_template.xml')  # 加载xml

# 测试一下
[[i.tag, i.text] for i in template.find('robot_info')]

import xml.etree.ElementTree as et
from config import *
from random import choice
import re


class template():
    def __init__(self):
        self.template = et.parse(TEMPLATE_PATH)
        self.robot_info = self.load_robot_info()  # 加载个人属性
        self.temp = self.template.findall('temp')  # 加载问答样式

    def load_robot_info(self):
        rebot_info = self.template.find('robot_info')
        rebot_info_dict = {}
        for info in rebot_info:
            rebot_info_dict[info.tag] = info.text
        return rebot_info_dict

    def search_answer(self, question):
        keys, temp = False, None
        for i in self.temp:
            qs = i.find('question')
            for q in qs.findall('q'):
                if re.search(q.text, question):
                    keys = True
                    temp = i
                    break
            if keys:
                return choice([i.text for i in temp.find('answer').findall('a')]).format(**self.robot_info)
        return None


test = template()
while 1:
    q = input('请输入->')
    if q == ':':
        break
    print(test.search_answer(q))