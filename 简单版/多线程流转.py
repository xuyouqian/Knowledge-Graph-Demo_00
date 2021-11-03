import time
import requests
import json
import _thread

question, answer = [None, None, None, None], [None, None, None, None]


def test_template():
    global question, answer
    while 1:
        if question[0] is not None:
            data = {
                'question': question[0]
            }
            url = 'http://127.0.0.1:8080/template'
            res = requests.post(url, data=json.dumps(data)).json()
            if question[0] is not None:
                answer[0] = res
                question[0] = None
            time.sleep(1)


def test_CorpusSearch():
    global question, answer
    while 1:
        if question[1] is not None:
            data = {
                'question': question[1]
            }
            url = 'http://127.0.0.1:8080/CorpusSearch'
            res = requests.post(url, data=json.dumps(data)).json()
            if question[1] is not None:
                answer[1] = res
                question[1] = None
            time.sleep(1)


def test_GraphSearch():
    global question, answer
    while 1:
        if question[2] is not None:
            data = {
                'question': question[2]
            }
            url = 'http://127.0.0.1:8080/GraphSearch'
            res = requests.post(url, data=json.dumps(data)).json()
            if question[2] is not None:
                answer[2] = res
                question[2] = None
            time.sleep(1)


def test_InterNet():
    global question, answer
    while 1:
        if question[3] is not None:
            data = {
                'question': question[3]
            }
            url = 'http://127.0.0.1:8080/InterNet'
            res = requests.post(url, data=json.dumps(data)).json()
            if question[3] is not None:
                answer[3] = res
                question[3] = None
            time.sleep(1)


from mychatbot import *
import xml.etree.ElementTree as et
from config import *


def get_default():
    answers = []
    data = et.parse(TEMPLATE_PATH)
    a = data.find('default').findall('a')
    for i in a:
        answers.append(i.text)
    return answers


if __name__ == '__main__':
    default_answer = get_default()
    _thread.start_new_thread(test_template, ())
    _thread.start_new_thread(test_CorpusSearch, ())
    _thread.start_new_thread(test_GraphSearch, ())
    _thread.start_new_thread(test_InterNet, ())
    while 1:
        answer = [None, None, None, None]
        question = [None, None, None, None]
        get = input('请输入->')
        if get == ':':
            break
        if get != '':
            question = [get, get, get, get]
        if question[0] is not None:
            n = 0
            while 1:
                k = False
                for i in answer:
                    if i is not None:
                        k = True
                        print(i)
                        break
                if k:
                    break
                time.sleep(1)
                n += 1
                if n > 10:
                    print(choice(default_answer))
                    break
