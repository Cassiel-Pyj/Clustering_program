#coding:utf-8
#_author_='PYJ'

#创建一个数据点，attributes代表数据集的一行，可含有多个属性，是个列表，clusterMark簇标签，标记它属于哪个簇,有密度值ro,与最近密度更大的邻居的距离dendist,该邻居neighbor
class Point:
    def __init__(self,attributes,clusterMark,ro,dendist,gamma,neighbor):
        self.attributes=attributes
        self.clusterMark=clusterMark
        self.ro=ro
        self.dendist=dendist
        self.gamma=gamma
        self.neighbor=neighbor
