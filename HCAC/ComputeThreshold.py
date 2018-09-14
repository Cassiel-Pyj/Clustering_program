#coding:utf-8
#_author_='PYJ'

from Point import *
import main
from Distance import *

# 找距离最小的元素对
def FindMinDist(disMat):
    mindist=np.inf
    x=0;y=0
    m=np.shape(disMat)[0]
    for i in range(0,m,1):
        for j in range(i+1,m,1):
            if mindist>disMat.tolist()[i][j] and disMat.tolist()[i][j]!=0:
                mindist=disMat.tolist()[i][j]
                x=i;y=j
    return x,y,mindist

# 找距离最小的元素对附近的第二小的元素对
def FindSecMinDist(x,y,disMat):
    secMindist=np.inf
    r=0;s=0
    m=np.shape(disMat)[0]
    for i in range(0,m,1):
        for j in range(i+1,m,1):
            if ((secMindist>disMat.tolist()[i][j] and i!=j) and (((i==x and j!=y) or (j==x and i!=y)) or ((i==y and j!=x) or (j==y and i!=x)))):
                secMindist=disMat.tolist()[i][j]
                r=i;s=j
    return r,s,secMindist

#dist_avg  簇间的距离：均链
def dist_avg(ptdata,Ci, Cj):
    # print('Ci',Ci)
    # print('Cj',Cj)
    if Ci==Cj:
        return 0
    n=len(ptdata)
    ptsInClusti=[]
    ptsInClustj=[]
    for i in range(n):
        if Ci==ptdata[i].clusterMark:
            ptsInClusti.append(ptdata[i])
        if Cj==ptdata[i].clusterMark:
            ptsInClustj.append(ptdata[i])
    # print('ptsInClusti',ptsInClusti)
    # print('ptsInClustj',ptsInClustj)
    leni=len(ptsInClusti)
    lenj=len(ptsInClustj)
    maxlen=max(leni,lenj)
    sumij=0
    for i in range(leni):
        for j in range(lenj):
            distij=Distance(ptsInClusti[i],ptsInClustj[j]).dis
            sumij+=distij
    avgdist=sumij/maxlen
    # print('Ci,Cj簇间平均距离',avgdist)
    return avgdist

# 计算信心值C的阈值confT,通过无监督凝聚层次聚类学习得到
def ComputeThreshold(n,q):
    # 初始化步骤
    C=[]    # 信心值备选阈值列表
    datalist=main.dataset.tolist()
    ptdata=[]                 # 存放n个数据点对象的列表ptdata，初始每个数据点是一个簇
    for i in range(n):
        ptdata.append(Point(datalist[i][0],datalist[i][1],i))
    disMat=np.mat(np.zeros((n,n))).tolist()  # 初始化距离矩阵,存放n个数据点对象之间的距离
    for i in range(n):
        for j in range(i,n):
            disMat[i][j]=Distance(ptdata[i],ptdata[j]).dis
    disMat=np.mat(disMat)
    # print('初始化距离矩阵如下\n', disMat)
    iter=n

    # 在无监督的凝聚层次聚类中，最多有n-1次合并
    for k in range(0,n-1):
        #-----------迭代地将两个最近的簇合并，直到当前聚类数目=K--------------
        while iter>main.K:
            #--------每次首先找距离最小的两个元素,再计算该元素对附近的最小距离的元素--------
            x,y,minDistk=FindMinDist(disMat)
            # print('簇的距离矩阵横坐标簇x:',x,'簇的距离矩阵纵坐标簇y:',y,'最小距离的元素对：',minDistk)
            r,s,secMinDistk=FindSecMinDist(x,y,disMat)
            # print('簇的距离矩阵横坐标簇r:',r,'簇的距离矩阵纵坐标簇s:',s,'次小距离的元素对：',secMinDistk)
            Ck=secMinDistk-minDistk
            # print('信心值：', Ck)
            C.append(Ck)
            # 初始化clusterMarkList
            clusterMarkList_or=[]
            for i in range(n):
                if ptdata[i].clusterMark not in clusterMarkList_or:
                    clusterMarkList_or.append(ptdata[i].clusterMark)
            # print('clusterMarkList_or:\n',clusterMarkList_or)
            #------------合并两个簇--------------
            smallerindex=min(clusterMarkList_or[x],clusterMarkList_or[y])
            biggerindex=max(clusterMarkList_or[x],clusterMarkList_or[y])
            for i in range(n):
                if ptdata[i].clusterMark==biggerindex:
                    ptdata[i].clusterMark=smallerindex
                # print('中间产生的簇分配',ptdata[i].clusterMark)
            iter=iter-1
            # 获取合簇后的clusterMarkList
            clusterMarkList=[]
            for i in range(n):
                if ptdata[i].clusterMark not in clusterMarkList:
                    clusterMarkList.append(ptdata[i].clusterMark)
            # print('clusterMarkList:\n',clusterMarkList)
            #----------更新距离矩阵disMat------------------
            disMat=np.mat(np.zeros((iter,iter))).tolist()
            for i in range(iter):
                for j in range(i+1,iter): #更新距离矩阵，用簇之间的距离计算
                    Ci=clusterMarkList[i]
                    Cj=clusterMarkList[j]
                    disMat[i][j]=dist_avg(ptdata,Ci,Cj)
            disMat=np.mat(disMat)
            # print('中间产生的距离矩阵',disMat)
    print("最终簇划分:")
    for i in range(n):
        print(ptdata[i].clusterMark)
    C.sort()
    print('C列表：', C)
    confT=C[q-1]
    print('最后算出来的阈值',confT)
    return confT









