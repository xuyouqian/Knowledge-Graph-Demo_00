# service
from flask_cors import cross_origin
from flask import Flask, request, redirect, url_for
import requests, json
from mychatbot import template, CorpusSearch, GraphSearch, InterNet

# global
app = Flask(__name__)
# init the chatbot
template_model = template()
CorpusSearch_model = CorpusSearch()
GraphSearch_model = GraphSearch()
InterNet_model = InterNet()


@app.route('/test', methods=['GET', 'POST'])
@cross_origin()
def myfirst_service():
    if request.method == "POST":
        # sta_post = time.time()
        data = request.get_data().decode()
        data = json.loads(data)
        return json.dumps('1', ensure_ascii=False)


@app.route('/template', methods=['GET', 'POST'])
@cross_origin()
def test_template():
    if request.method == "POST":
        # sta_post = time.time()
        data = request.get_data().decode()
        data = json.loads(data)
        question = data['question']
        answer = template_model.search_answer(question)
        return json.dumps(answer, ensure_ascii=False)


@app.route('/CorpusSearch', methods=['GET', 'POST'])
@cross_origin()
def test_CorpusSearch():
    if request.method == "POST":
        # sta_post = time.time()
        data = request.get_data().decode()
        data = json.loads(data)
        question = data['question']
        answer = CorpusSearch_model.search_answer(question)
        return json.dumps(answer, ensure_ascii=False)


@app.route('/GraphSearch', methods=['GET', 'POST'])
@cross_origin()
def test_GraphSearch():
    if request.method == "POST":
        # sta_post = time.time()
        data = request.get_data().decode()
        data = json.loads(data)
        question = data['question']
        answer = GraphSearch_model.search_answer(question)
        return json.dumps(answer, ensure_ascii=False)


@app.route('/InterNet', methods=['GET', 'POST'])
@cross_origin()
def test_InterNet():
    if request.method == "POST":
        # sta_post = time.time()
        data = request.get_data().decode()
        data = json.loads(data)
        question = data['question']
        if '是谁' in question or '关系' in question:
            return json.dumps(None, ensure_ascii=False)
        try:
            answer = InterNet_model.search_answer(question)
        except:
            answer = None
        # except:
        # answer = '对不起啊，小智无法解决这个问题'
        return json.dumps(answer, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, threaded=True)
