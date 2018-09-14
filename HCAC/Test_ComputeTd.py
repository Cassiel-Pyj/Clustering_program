#coding:utf-8
#_author_='PYJ'

from Point import *
from main import *
from Distance import *

# 计算簇间距离
def getDistance(cluster1, cluster2, disList, ptdata):
    distance = 0
    cluster1_num = 0
    cluster2_num = 0
    # 求总距离长度
    for i in range(len(disList)):
        if disList[i].xpt.clusterMark == cluster1 and disList[i].ypt.clusterMark == cluster2 or disList[i].xpt.clusterMark == cluster2 and disList[i].ypt.clusterMark == cluster1:
            distance += disList[i].dis
    # 求较大簇中元素数
    for i in range(len(ptdata)):
        if ptdata[i].clclusterMark == cluster1:
           cluster1_num += 1
        elif ptdata[i].clclusterMark == cluster2:
           cluster2_num += 1
    return distance/max(cluster1_num, cluster2_num)

# 计算信心值C的阈值confT,通过无监督凝聚层次聚类学习得到
def ComputeTd(n,q):
    # 初始化步骤
    C = []    # 信心值备选阈值列表
    datalist = dataset.tolist()
    ptdata = []                 # 存放n个数据点对象的列表ptdata，初始每个数据点是一个簇
    disList = []                # 距离实例列表
    print(datalist)
    # 点列表赋值
    for i in range(0, n):
        ptdata.append(Point(datalist[i][0], datalist[i][1], i))
    # 距离列表赋值
    for i in range(0, len(dataset) - 1):
        for j in range(i+1 ,n):
            disList.append(Distance(ptdata[i], ptdata[j]))
    # 距离列表选择小到大排序
    for i in range(0, len(disList)-1):
        for j in range(i+1 ,len(disList)):
            if disList[i].dis > disList[j].dis:
                disK = disList[i]
                disList[i] = disList[j]
                disList[j] = disK
    print('初始化距离列表如下:')
    for i in range(len(disList)):
        print([disList[i].xpt.xlabel, disList[i].xpt.ylabel], [disList[i].ypt.xlabel, disList[i].ypt.ylabel], disList[i].dis)
    iter=n  # 簇数目
    # 在无监督的凝聚层次聚类中，最多有n-1次合并
    for k in range(0,n-1):
        #-----------迭代地将两个最近的簇合并，直到当前聚类数目=K--------------
        while iter>K:
            #--------每次首先找距离最小的两个元素,再计算该元素对附近的最小距离的元素--------
            # 查询并记录已有分簇
            clusterMarkList = []
            for i in range(len(ptdata)):
                 if ptdata[i].clusterMark not in clusterMarkList:
                    clusterMarkList.append(ptdata[i].clusterMark)
            print(clusterMarkList)
            # 找最短距离(非同簇)
            minDistk = 0
            for i in range(0, len(clusterMarkList)-1):
                for j in range(i+1, len(clusterMarkList)):
                    if i == 0:
                        x = disList[i].xpt
                        y = disList[i].ypt
                        minDistk = getDistance(clusterMarkList[i], clusterMarkList[j], disList, ptdata)
                    else:
                        distance = getDistance(clusterMarkList[i], clusterMarkList[j], disList, ptdata)
                        if minDistk > distance :
                            minDistk = distance
            print('最短元素距离：',  minDistk)
            # 找次短距离(非同簇and不是最近距离元素对)
            for i in range(len(disList)):
                if (disList[i].xpt.clusterMark != disList[i].ypt.clusterMark) and ((disList[i].xpt.clusterMark == x.clusterMark and disList[i].ypt.clusterMark != y.clusterMark) or(disList[i].xpt.clusterMark == y.clusterMark and disList[i].ypt.clusterMark != x.clusterMark) or (disList[i].ypt.clusterMark == x.clusterMark and disList[i].xpt.clusterMark != y.clusterMark) or (disList[i].ypt.clusterMark == y.clusterMark and disList[i].xpt.clusterMark != x.clusterMark)):
                    r = disList[i].xpt
                    s = disList[i].ypt
                    secMinDistk = disList[i].dis
                    break
            print('次短元素距离：', [r.xlabel, r.ylabel, r.clusterMark], [s.xlabel, s.ylabel,s.clusterMark], secMinDistk)
            # 计算信心值
            Ck=secMinDistk-minDistk
            print(Ck)
            C.append(Ck)
            #------------合并两个簇--------------
            smallerindex=min(x.clusterMark, y.clusterMark)
            biggerindex=max(x.clusterMark, y.clusterMark)
            for i in range(n):
                if ptdata[i].clusterMark==biggerindex:
                    ptdata[i].clusterMark=smallerindex
                print('中间产生的簇分配',ptdata[i].clusterMark)
            iter=iter-1
    print("最终簇划分:")
    for i in range(n):
        print(ptdata[i].clusterMark)
    print(C)
    C.sort()
    print('C列表：', C)
    confT=C[q-1]
    print('最后算出来的阈值',confT)
    return confT

