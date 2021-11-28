from py2neo import Graph
from pyhanlp import *


class GraphSearch:
    def __init__(self):
        self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")

    # 基于科目，寻求课程
    def forintent0(self, entity):
        sql = "match p = (n:%s)-[]->(m:km2) where n.name = '%s' return m.name" % (entity[1], entity[0])
        res = self.graph.run(sql).data()
        return [i['m.name'] for i in res]

    def forintent1(self, entity):
        if 1:
            if entity[1] == 'km1':
                sql = "match p = (n:%s)-[]->()-[]->(m:kg) where n.name = '%s' return m.name" % (entity[1], entity[0])
            else:
                sql = "match p = (n:%s)-[]->(m:kg) where n.name = '%s' return m.name" % (entity[1], entity[0])
            res = self.graph.run(sql).data()
            return [i['m.name'] for i in res]

    # 基于课程、科目、知识点，寻求问题
    def forintent2(self, entity):
        if 1:
            if entity[1] == 'km1':
                sql = "match p = (n:%s)-[]->()-[]->()-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            elif entity[1] == 'km2':
                sql = "match p = (n:%s)-[]->()-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            else:
                sql = "match p = (n:%s)-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            res = self.graph.run(sql).data()
            return [i['m.name'] for i in res]

    # 基于科目或课程，寻求知识点
    def forintent4(self, entity):
        if 1:
            if entity[1] == 'km1':
                sql = "match p = (n:%s)-[]->()-[]->(m:kg) where n.name = '%s' return m.name" % (entity[1], entity[0])
            else:
                sql = "match p = (n:%s)-[]->(m:kg) where n.name = '%s' return m.name" % (entity[1], entity[0])
            res = self.graph.run(sql).data()
            return [i['m.name'] for i in res]

    # 基于课程、科目、知识点，寻求问题
    def forintent5(self, entity):
        if 1:
            if entity[1] == 'km1':
                sql = "match p = (n:%s)-[]->()-[]->()-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            elif entity[1] == 'km2':
                sql = "match p = (n:%s)-[]->()-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            else:
                sql = "match p = (n:%s)-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            res = self.graph.run(sql).data()
            return [i['m.name'] for i in res]

    # 基于科目，寻求课程
    def forintent6(self, entity):
        if 1:
            sql = "match p = (n:km1)-[]->(m:%s) where m.name = '%s' return n.name limit 10" % (entity[1], entity[0])
            res = self.graph.run(sql).data()
            return [i['n.name'] for i in res]

    def forintent8(self, entity):
        if 1:
            if entity[1] == 'km1':
                sql = "match p = (n:%s)-[]->()-[]->()-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            elif entity[1] == 'km2':
                sql = "match p = (n:%s)-[]->()-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            else:
                sql = "match p = (n:%s)-[]->(m:question) where n.name = '%s' return m.name limit 10" % (
                entity[1], entity[0])
            res = self.graph.run(sql).data()
            return [i['m.name'] for i in res]

    # 基于作者，寻求诗句名
    def forintent9(self, entity):
        sql = "match p = (n:%s)-[]->(m:title) where n.name = '%s' return m.name" % (entity[1], entity[0])
        print(sql)
        res = self.graph.run(sql).data()
        print(res)
        return [i['m.name'] for i in res]

    # 基于作者名，寻求简介
    def forintent10(self, entity):
        sql = "match p = (n:%s)-[]->(m:introduce) where n.name = '%s' return m.name" % (entity[1], entity[0])
        res = self.graph.run(sql).data()
        return res[0]['m.name']

    def forintent11(self, entity):
        if 1:
            sql = "match p = (n:author)-[]->(m:title) where m.name = '%s' return n.name" % (entity[0])
            res = self.graph.run(sql).data()
            return [i['n.name'] for i in res]

    # 基于诗名，寻求诗文
    def forintent12(self, entity):
        if 1:
            sql = "match p = (n:title)-[]->(m:content) where n.name = '%s' return m.name" % (entity[0])
            res = self.graph.run(sql).data()
            return [i['m.name'] for i in res]

    # 基于诗名，寻求翻译
    def forintent13(self, entity):
        if 1:
            sql = "match p = (n:title)-[]->(m:translate) where n.name = '%s' return m.name" % (entity[0])
            res = self.graph.run(sql).data()
            return [i['m.name'] for i in res]

    # 基于诗名，寻求类型
    def forintent14(self, entity):
        if 1:
            sql = "match p = (n:title)<-[]-(m:tag) where n.name = '%s' return m.name" % (entity[0])
            res = self.graph.run(sql).data()
            return [i['m.name'] for i in res]
