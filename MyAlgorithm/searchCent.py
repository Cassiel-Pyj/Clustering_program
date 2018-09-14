#coding:utf-8
#_author_='PYJ'

import math
import numpy as np
from scipy.linalg import norm
from MyAlgorithm.Point import *
from MyAlgorithm.Plot import showGamma
from computeDC import *

#计算欧氏距离(cut-off kernel使用)
def distEuclDis(vecA,vecB):
    # return sqrt(sum(power(vecA-vecB,2)))
    return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数

#两个向量的核化，采用高斯核，sigma需自己取值
def Kernel(vecA,vecB):
    sigma=0.5
    kernel=math.exp((-1*norm(vecA-vecB)/(2*math.pow(sigma,2))))
    return kernel

# 计算每个点i的密度ro,有cut-off Kernel(离散值)
def ComputeRO(dataset,pti,dc,dist=Kernel):
    ro=0
    for ptj in dataset.tolist():
        if dist(np.array(pti),np.array(ptj))<dc:
            ro+=1
    return ro

#计算每个点i的密度ro,使用高斯核计算（连续值）
def ComputeRO2(dataset,pti,dc,dist=Kernel):
    ro=0
    for ptj in dataset:
        ro+=math.exp(-1*math.pow(dist(pti,ptj)/dc,2))
    return ro

#计算每个点i的最近密度更大的邻居，与它的距离dendist
def ComputeDendist(ptdata,pti,dist=Kernel):
    # print("pti",pti)
    mindendist=np.inf
    for ptj in range(pti):
        h=dist(ptdata[pti].attributes,ptdata[ptj].attributes)
        # print("h距离",h)
        if mindendist>h:
            mindendist=h
    return mindendist

def searchCent(ptdata,k):
    gammaList=[]
    for p in ptdata:
        p.gamma=p.ro*p.dendist
        gammaList.append(p.gamma)
    # print(gammaList)
    gammaList.sort(reverse=True)
    # print("从大到小排序好的Gamma",gammaList)
    # showGamma(gammaList)  #把Gamma图画出来
    gammaList=gammaList[:k]
    # print("选择前K个最大的gamma",gammaList)
    centList=[]
    i=0
    for p in ptdata:
        if p.gamma in gammaList:
            centList.append(p)
            p.clusterMark=i
            i+=1
    # print("得到的中心点K个",centList)
    return centList

def beginSearch(dataset,K,t):
    n=np.shape(dataset)[0]
    ptdata=[]                 # 存放n个数据点对象的列表ptdata
    # 初始化与预处理（确定截断距离dc）
    dc=ComputeDC(dataset,t)
    # 对于每一个数据点，计算其密度ro,使用高斯核或者cut-off核，并降序排序
    for pt in dataset:
        ro=ComputeRO2(dataset,pt,dc)
        ptdata.append(Point(pt,-1,ro,-1,-1))
    for i in range(n): #根据密度大小对点对象列表进行冒泡排序，结果的ptdata按密度降序排序
        current_status = False
        for j in range(n-i-1):
             if ptdata[j].ro < ptdata[j+1].ro:
                 ptdata[j], ptdata[j+1] = ptdata[j+1], ptdata[j]
                 current_status = True
        if not current_status:
             break
    # 对每一个数据点，计算其密度距离dendist,记录比它密度更大的最近邻
    for i in range(1,n):
        ptdata[i].dendist=ComputeDendist(ptdata,i)
    # print([x.dendist for x in ptdata])
    ptdata[0].dendist=max([x.dendist for x in ptdata])
    centList=searchCent(ptdata,K)
    cent=[]
    for ce in centList:
        cent.append(ce.attributes.tolist()[0])
    return cent,dc



