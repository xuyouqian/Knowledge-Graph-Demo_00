import requests
import json

# data = {
#    'test':'你好啊我的第一次服务'
#   }

# url = 'http://127.0.0.1:8080/test'
# r = requests.post(url,data=json.dumps(data))
# print(r.json())

while 1:
    question = input('请输入->')
    data = {
        'question': question
    }
    url = 'http://127.0.0.1:8080/template'
    answer = requests.post(url, data=json.dumps(data)).json()
    if answer is not None:
        print(answer)
        continue
    url = 'http://127.0.0.1:8080/CorpusSearch'
    answer = requests.post(url, data=json.dumps(data)).json()
    if answer is not None:
        print(answer)
        continue
    print('请稍等哦~我正在尽力搜索我的记忆ing~')
    url = 'http://127.0.0.1:8080/GraphSearch'
    answer = requests.post(url, data=json.dumps(data)).json()
    if answer is not None:
        print(answer)
        continue
    print('小智发现我的记忆力缺少了点内容，o(╥﹏╥)o，让我查查网上怎么说~')
    url = 'http://127.0.0.1:8080/InterNet'
    answer = requests.post(url, data=json.dumps(data)).json()
    print(answer)
