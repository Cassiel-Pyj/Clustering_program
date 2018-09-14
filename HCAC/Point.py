#coding:utf-8
#_author_='PYJ'

#创建一个数据点，每个数据点有一个簇标签，标记它属于哪个簇
class Point:
    def __init__(self,xlabel,ylabel,clusterMark):
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.clusterMark=clusterMark