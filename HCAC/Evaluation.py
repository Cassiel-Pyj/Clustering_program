#coding:utf-8
#_author_='PYJ'
import numpy as np
import main

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

def  ContingenceTable(ptdata,answerClusterMark):
    #生成一个列联表
    markList=[]
    for i in range(main.n):
        if ptdata[i].clusterMark not in markList:
            markList.append(ptdata[i].clusterMark)
    Table =[[0 for col in range(len(answerClusterMark))] for row in range(len(markList))]
    for i in range(len(markList)):
        for j in range(len(answerClusterMark)):
            Table[i][j]=computeTable(markList[i],j,answerClusterMark,ptdata)
    print('当前簇对的列联表Table\n',Table)
    return Table

def OuterEvaluation(ptdata,answerClusterMark,K):
   # 计算F度量，值越大越好
    contingenceTable=ContingenceTable(ptdata,answerClusterMark)
    contingenceTable=np.mat(contingenceTable)
    print(contingenceTable)
    F_sum=0
    for i in range(K):
        preci=max(contingenceTable.tolist()[i])/sum(contingenceTable.tolist()[i]) #先计算精度
        recalli=max(contingenceTable.tolist()[i])/sum(contingenceTable[:,i])  #再计算召回率
        Fi=2*preci*recalli/(preci+recalli)  #F度量是精度和召回率值得调和平均数
        F_sum+=Fi
    F_Measure=float(F_sum/K)
    return  F_Measure



