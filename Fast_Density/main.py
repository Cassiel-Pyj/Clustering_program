#coding:utf-8
#_author_='PYJ'
import time
import math
import numpy as np
from Fast_Density.LoadData import createData,readFile,readUCI,doPCA
from Fast_Density.Point import Point
from Fast_Density.ComputeDC import *
from scipy.linalg import norm
from SearchCent import *
from Fast_Density.showPlot import *

# 计算每个点i的密度ro,cut-off Kernel（离散值）
def ComputeRO(dataset,pti,dc,dist=distEuclDis):
    ro=0
    for ptj in dataset:
        if dist(pti,ptj)<dc:
            ro+=1
    return ro

#计算每个点i的密度ro,使用高斯核计算（连续值）
def ComputeRO2(dataset,pti,dc,dist=distEuclDis):
    ro=0
    for ptj in dataset:
        ro+=math.exp(-1*math.pow(dist(pti,ptj)/dc,2))
    return ro

#计算每个点i的最近密度更大的邻居，与它的距离dendist，记录该邻居
def ComputeDendist(ptdata,pti,dist=distEuclDis):
    # print("pti",pti)
    mindendist=np.inf
    neibor=Point(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf)
    for ptj in range(pti):
        h=dist(ptdata[pti].attributes,ptdata[ptj].attributes)
        # print("h距离",h)
        if mindendist>h:
            mindendist=h
            neibor=ptdata[ptj]
    return mindendist,neibor

# 快速密度聚类算法
def FastDensity(dataset,t):
    n=np.shape(dataset)[0]
    ptdata=[]                 # 存放n个数据点对象的列表ptdata

    # 初始化与预处理（确定截断距离dc）
    dc=ComputeDC(dataset,t)

    # 对于每一个数据点，计算其密度ro,使用高斯核或者cut-off核，并降序排序
    for pt in dataset:
        ro=ComputeRO2(dataset,pt,dc)
        ptdata.append(Point(pt,-1,ro,-1,-1,-1))
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
        ptdata[i].dendist,ptdata[i].neighbor=ComputeDendist(ptdata,i)
    ptdata[0].dendist=max([x.dendist for x in ptdata])
    ptdata[0].neighbor=ptdata[0]
    for p in ptdata:
        print("每个点的信息，密度,密度距离,密度更大最近邻",p.attributes,[p.ro,p.dendist],p.neighbor.attributes)

     # 画出决策图,直观地找到簇中心点
    PlotDecisionGraph(ptdata)

    # 使用密度*距离的值确定聚类中心
    centList=searchCent(ptdata)
    for ce in centList:
        print("选定的中心点：",ce.attributes,[ce.ro,ce.dendist],ce.neighbor,ce.clusterMark)

    # 对非聚类中心的点进行簇分配,每个点与密度更大的最近邻（距离小于dc）组成同一个簇
    resultClusterAssement=[]
    for pt in ptdata:
        if pt.clusterMark==-1:
            pt.clusterMark=pt.neighbor.clusterMark
        resultClusterAssement.append(pt.clusterMark)
        # print("每个点的簇分配结果",[pt.xlabel,pt.ylabel],pt.clusterMark)

    # # 噪声点检测，每一个cluster 中的点，分为cluster core 点和cluster halo点
    # --------为每一个簇计算平均局部密度最大值pb,该界限用来区分cluster halo--------
    pb=[0 for x in range(len(centList))]
    for i in range(len(ptdata)-1):
        for j in range(i+1,len(ptdata)):
            if ptdata[i].clusterMark!=ptdata[j].clusterMark and distEuclDis(ptdata[i].attributes,ptdata[j].attributes)<dc:
                roaverage=0.5*(ptdata[i].ro+ptdata[j].ro)
                if roaverage>pb[ptdata[i].clusterMark]:
                    pb[ptdata[i].clusterMark]=roaverage
                if roaverage>pb[ptdata[j].clusterMark]:
                    pb[ptdata[j].clusterMark]=roaverage
    print("计算好的pb",pb)
    # -----------对于halo区域的点，把clusterMark标记为-2，为噪声点--------------
    for p in ptdata:
        if p.ro<pb[p.clusterMark]:
            p.clusterMark=-2
    return ptdata,centList

if __name__ == "__main__":
    data,answerMark=readFile("E:\毕业设计\毕业设计\DataSet\Spiral.txt")
    # print(answerMark) #文件中的簇分配标签，可能要结合最后的效果验证，改成answerList?
    # print(data)
    # data,answerMark=readUCI("E:\毕业设计\毕业设计\DataSet\Data_Seeds.csv")
    t=0.02
    # data=doPCA(data,2)  #先降维后聚类
    start_time=time.time()
    ptdata,centList=FastDensity(data,t)
    showPlot(ptdata,centList)  #人造数据集的画图

    # showPCAplot(ptdata,centList) #UCI数据集的画图
    # showTruePlot(data,answerMark)
    end_time=time.time()
    costTime=end_time-start_time