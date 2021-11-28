import requests


class InterNet():
    def __init__(self):
        pass

    def search_answer(self, question):
        url = 'https://api.ownthink.com/bot?appid=xiaosi&userid=user&spoken='
        try:
            text = requests.post(url + question).json()
            if 'message' in text and text['message'] == 'success':
                return text['data']['info']['text']
            else:
                return None
        except:
            return None


test = InterNet()
while 1:
    question = input('请输入->')
    if question == ':':
        break
    print(test.search_answer(question))
