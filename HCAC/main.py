#coding:utf-8
#_author_='PYJ'

import time
from ComputeThreshold import *
from GeneratePool import *
from ComputeEntropy import *
from showPlot import *
from Evaluation import OuterEvaluation
from LoadData import readFile,readUCI

#全局变量
dataset,answerClusterMark=readFile("E:\毕业设计\毕业设计\DataSet\\artificial_datasets\\artificial_5.data")
n=np.shape(dataset)[0]
K=5
sizeOfPool=5  #簇对池的大小

if __name__ == "__main__":
    # 合并当前簇对的操作
    def doMerge(clusterMark1,clusterMark2,ptdata):
        smallerindex=min(clusterMark1,clusterMark2)
        biggerindex=max(clusterMark1,clusterMark2)
        for i in range(n):
            if ptdata[i].clusterMark==biggerindex:
                ptdata[i].clusterMark=smallerindex
            print('中间产生的簇分配',ptdata[i].clusterMark)
        # 更新合簇后的clusterMarkList
        clusterMarkList=[]
        for i in range(n):
            if ptdata[i].clusterMark not in clusterMarkList:
                clusterMarkList.append(ptdata[i].clusterMark)
        print('clusterMarkList:\n',clusterMarkList)
        return clusterMarkList

    # 更新簇的距离矩阵的操作
    def newdisMat(clusterMarkList,iter,ptdata):
        disMat=np.mat(np.zeros((iter,iter))).tolist()
        for i in range(iter):
            for j in range(i+1,iter): #更新距离矩阵，用簇之间的距离计算
                Ci=clusterMarkList[i]
                Cj=clusterMarkList[j]
                disMat[i][j]=dist_avg(ptdata,Ci,Cj)
        disMat=np.mat(disMat)
        print('中间产生的距离矩阵',disMat)
        return disMat

    #--------开始HCAC算法主程序------------
    start_time=time.time()
    interventions=int((n-1)*0.01)  #人为干预次数的上限,设定为数据集个数n-1的1%，5%，10%....100%；
    confT=ComputeThreshold(n,interventions) #获取信心值C的阈值confT,与簇标记答案
    # -------------初始化步骤---------------------
    C=0 #信心值
    datalist=dataset.tolist()
    ptdata=[]                 # 存放n个数据点对象的列表ptdata，初始每个数据点是一个簇
    for i in range(n):
        ptdata.append(Point(datalist[i][0],datalist[i][1],i))
    disMat=np.mat(np.zeros((n,n))).tolist()  # 初始化距离矩阵,存放n个数据点对象之间的距离
    for i in range(n):
        for j in range(i,n):
            disMat[i][j]=Distance(ptdata[i],ptdata[j]).dis
    disMat=np.mat(disMat)
    print('初始化距离矩阵如下\n', disMat)
    iter=n    #簇的个数初始化是所有数据点的个数n
    # -----------迭代地将两个最近的簇合并，直到当前聚类数目=K------------
    while iter>K:
        #--------每次首先找距离最小的元素对,再计算该元素对附近的最小距离的元素--------
        x,y,minDistk=FindMinDist(disMat)
        print('簇的距离矩阵横坐标簇x:',x,'簇的距离矩阵纵坐标簇y:',y,'最小距离的元素对：',minDistk)
        r,s,secMinDistk=FindSecMinDist(x,y,disMat)
        print('簇的距离矩阵横坐标簇r:',r,'簇的距离矩阵纵坐标簇s:',s,'次小距离的元素对：',secMinDistk)
        C=secMinDistk-minDistk
        print('信心值：', C)
        #--------判断当前信心值是否小于阈值，是，则进入外部询问（人为干预阶段），否，则信心值大，直接合并该簇对-------------
        # 初始化clusterMarkList
        clusterMarkList_or=[]
        for i in range(n):
            if ptdata[i].clusterMark not in clusterMarkList_or:
                clusterMarkList_or.append(ptdata[i].clusterMark)
        print('clusterMarkList_or:\n',clusterMarkList_or)
        # 若信心值低，进入人工干预阶段，生成sizeOfPool对簇对池
        if C<confT:
            print('当前将要合并的簇标号2个:\n',clusterMarkList_or[x],clusterMarkList_or[y])
            P=GeneratePool(n,sizeOfPool,x,y,disMat)
            bestPair=ComputeEntropy(P,ptdata,answerClusterMark)
            print("熵值最小的元素对为：\n",bestPair)
            #------------合并选出来的最好的簇对-------
            clusterMarkList=doMerge(bestPair[0],bestPair[1],ptdata)
            iter=iter-1
            #----------更新距离矩阵disMat------------------
            disMat=newdisMat(clusterMarkList,iter,ptdata)
        # 若信心值高
        else:
            #------------直接合并两个簇--------------
            clusterMarkList=doMerge(clusterMarkList_or[x],clusterMarkList_or[y],ptdata)
            iter=iter-1
            #----------更新距离矩阵disMat------------------
            disMat=newdisMat(clusterMarkList,iter,ptdata)
    print("最终簇划分:")
    resultClusterAssement=[]
    for i in range(n):
        resultClusterAssement.append(ptdata[i].clusterMark)
    print(resultClusterAssement)
    FMeasure=OuterEvaluation(ptdata,answerClusterMark,K)
    print("F度量:\n",FMeasure)
    end_time = time.time()
    print('所花时间：',end_time - start_time)
    showPlot(resultClusterAssement)
