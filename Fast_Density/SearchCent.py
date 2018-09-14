#coding:utf-8
#_author_='PYJ'

from sklearn.preprocessing import Normalizer
import numpy as np
from Fast_Density.showPlot import showGamma

def searchCent(ptdata):
    gammaList=[]
    for p in ptdata:
        p.gamma=p.ro*p.dendist
        gammaList.append(p.gamma)
    print(gammaList)
    gammaList.sort(reverse=True)
    print("从大到小排序好的Gamma",gammaList)
    showGamma(gammaList)
    k=3
    gammaList=gammaList[:k]
    print("选择前K个最大的gamma",gammaList)
    centList=[]
    i=0
    for p in ptdata:
        if p.gamma in gammaList:
            centList.append(p)
            p.clusterMark=i
            i+=1
    # print("得到的中心点K个",centList)
    return centList


