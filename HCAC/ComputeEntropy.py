#coding:utf-8
#_author_='PYJ'

import main
import math
import numpy as np

def computeTable(clusteri,classj,answerClusterMark,ptdata):
    ptsInClusteri=[]
    for i in range(main.n):
        if ptdata[i].clusterMark==clusteri:
            ptsInClusteri.append([ptdata[i].xlabel,ptdata[i].ylabel])
    # print("ptsInClusteri:\n",ptsInClusteri)
    num=0
    for pts in ptsInClusteri:
        if pts in answerClusterMark[classj]:
                num+=1
    return num

# 首先获取当前簇对的列联表ContingenceTable
def  ContingenceTable(pair,ptdata,answerClusterMark):
    #对于每个簇对，生成一个列联表
    Table =[[0 for col in range(len(answerClusterMark))] for row in range(2)]
    for i in range(2):
        for j in range(len(answerClusterMark)):
            Table[i][j]=computeTable(pair[i],j,answerClusterMark,ptdata)
    print('当前簇对的列联表Table\n',Table)
    return Table

#根据列联表计算当前簇对的熵值
def entropyOfThisPair(contingenceTable,answerClusterMark):
    Entropy=0
    for i in range(2):
        Hi=0
        for j in range(len(answerClusterMark)):
            if contingenceTable[i][j]!=0:
                Hi+=-1*(contingenceTable[i][j]/sum(contingenceTable[i]))*math.log(contingenceTable[i][j]/sum(contingenceTable[i]),2)
        Entropy=Hi
    return Entropy

# 计算生成的簇对池中，每一对簇的熵值，找熵值最小的（聚类效果最好）的一对作为当前被询问时，用户选择的最应该被合并的簇对
def ComputeEntropy(P,ptdata,answerClusterMark):
    # ----首先生成外部类与实际聚类情况的列联表--------
    min_entropy=np.inf
    thePair=[]
    for elements in P:
        contingenceTable=ContingenceTable(elements,ptdata,answerClusterMark)
        entropy=entropyOfThisPair(contingenceTable,answerClusterMark)
        if min_entropy>entropy:
            min_entropy=entropy
            thePair=elements
    print("熵值最小为：\n",min_entropy)
    return thePair


