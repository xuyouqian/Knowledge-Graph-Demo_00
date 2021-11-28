from py2neo import Graph, NodeMatcher, RelationshipMatcher
from pyhanlp import *
from pyhanlp import HanLP
from random import choice


# 知识图谱的初步应用
class GraphSearch():
    def __init__(self):
        self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")
        self.iswho_sql = "profile match p=(n)<-[r]-(b) where n.name='%s' return n.name,r.name,b.name"
        self.isrelation_sql = "profile match p=(n)<-[r]-(b) where n.name=~'%s' and b.name=~'%s' return n.name,r.name,b.name"
        self.n_matcher = NodeMatcher(self.graph)
        self.r_matcher = RelationshipMatcher(self.graph)

    def search_answer(self, question):
        # 使用HanLP进行词性判断
        sentence = HanLP.parseDependency(question)
        # 后续可以替换成自己训练的模块首先针对句意进行分析，其次针对目标实体进行提取；但主要也是针对业务场景进行分析和处理
        seg = {}
        res_combine = ''
        for word in sentence.iterator():  # 通过dir()可以查看sentence的方法
            # print("%s --(%s)--> %s" % (word.LEMMA, word.DEPREL, word.HEAD.LEMMA))
            print("%s --(%s)--> %s--> %s--> %s--> %s--> %s" % (
                word.ID, word.LEMMA, word.CPOSTAG, word.POSTAG, word.HEAD.NAME, word.DEPREL, word.NAME))
        for word in sentence.iterator():
            ##只处理nr名词：人，v动词，n名词，针对进行提问进行词性分析
            if word.POSTAG[0] == 'n' or word.POSTAG in ['v', 'r']:
                if word.POSTAG not in seg:
                    seg[word.POSTAG] = [word.LEMMA]
                else:
                    seg[word.POSTAG].append(word.LEMMA)
        print(seg, '*' * 10)
        # 简单基于词性和内容判断是否为目标句式'A是谁'以此使用知识图谱进行回答
        if 'v' in seg and '是' in seg['v']:
            if 'r' in seg and 'nr' in seg and '谁' in seg['r']:
                for person in seg['nr']:
                    res = self.entity_is(person)
                    res_combine = []
                    for i in res[:10]:
                        res_combine.append('%s是:%s%s' % (i.end_node['name'], i.start_node['name'], i['name']))
                return choice(res_combine)
        # 基于词性和内容判断是否为目标句式'A和B的关系'以此使用知识图谱进行回答
        if 'n' in seg and '关系' in seg['n']:
            if len(seg['nr']) == 2:
                print(seg, '*1' * 10)
                res1 = self.graph.run(self.isrelation_sql % (seg['nr'][1], seg['nr'][0])).data()
                if res1 != []:
                    res_combine = seg['nr'][0] + '的' + res1[0]['r.name'] + '是' + seg['nr'][1]
                    return res_combine
                res2 = self.graph.run(self.isrelation_sql % (seg['nr'][0], seg['nr'][1])).data()
                if res2 != []:
                    res_combine = seg['nr'][1] + '的' + res2[0]['r.name'] + '是' + seg['nr'][0]
                    return res_combine
        if res_combine == '':
            return None

    def entity_is(self, name):
        node = self.n_matcher.match("ENTITY", name=name).first()
        relations = self.r_matcher.match(nodes=[None, node]).all()
        return relations


test = GraphSearch()
while 1:
    print(test.search_answer(input('请输入->')))
