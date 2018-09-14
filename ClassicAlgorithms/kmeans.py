#coding:utf-8
#_author_='PYJ'

from sklearn.preprocessing import Normalizer
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import norm
from sklearn.datasets import *
import random
import time
import math

#随机生成训练模型的数据
def createData():
    # rand_data = random.normal(0.5,10000,size=(150,2)) #numpy中的正态分布
    # x,y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2) #生成分组高斯混合数据
    # x,y=make_circles(n_samples=500,factor=0.2,noise=0.04)  #生成环形数据
    # x,y=make_moons(n_samples=300,noise=0.1)
    x,y = make_blobs(n_samples=150, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2]) #生成聚类模型数据
    # print('生成模型数据的质心:\n[[-1,-1], [1,1], [2,2]]\n')
    rand_data=x
    return rand_data

#创建K个随机质心的集合（基本K-means的方法）
def randCent(dataset,K):
    n=np.shape(dataset)[1] #数据集列的数量
    centroids=np.mat(np.zeros((K,n))) #初始化K行n列的质心矩阵
    for j in range(n):
        minJ=min(dataset[:,j])  #求数据集每一列的最小值、最大值
        maxJ=max(dataset[:,j])
        rangeJ=float(maxJ-minJ) #质心的取值范围在最大与最小值之间
        centroids[:,j]=np.mat(minJ+rangeJ*np.random.rand(K,1))
    return centroids

#创建K个随机质心的集合（K-means++的方法）
def randCentpp(dataset,K):
    m=np.shape(dataset)[0] #数据集行的数量
    n=np.shape(dataset)[1] #数据集列的数量
    centriods=[]
    cent0=[]
    #第一个质心随机取
    for j in range(n):
        minJ=min(dataset[:,j])  #求数据集每一列的最小值、最大值
        maxJ=max(dataset[:,j])
        rangeJ=float(maxJ-minJ)
        cent0.append(minJ+rangeJ*random.random())
    centriods.append(cent0)
    while len(centriods)<K:
        #计算每个点i到与其最近的质心j的距离
        weights=[]
        for i in range(m):
            mindist=np.inf
            for j in range(len(centriods)):
                distIJ=distEuclDis(dataset[i],centriods[j])
                if distIJ<mindist:
                    mindist=distIJ
            weights.append(mindist)
        # print('weights:\n',weights)
        total=sum(weights)
        #得到概率分布数组weights
        weights=[x/total for x in weights]
        num=random.random()
        summ=0;x=-1
        while summ<num:
            x+=1
            summ+=weights[x]
        centriods.append(dataset[x])
    return np.mat(centriods)

#创建K个随机质心的集合（从数据集中随机取K个)
def randCentNormal(dataset,K):
    centroids=np.mat(random.sample(dataset.tolist(),K))
    return centroids

#计算欧氏距离
def distEuclDis(vecA,vecB):
    # return sqrt(sum(power(vecA-vecB,2)))
    # print(vecA,vecB)
    return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数

#做聚类的验证，先获取数据集的列联表ContingenceTable
def  ContingenceTable(clusterAssment,m,K):
    groupnum = []
    # 查找组标识
    for i in range(m):
        if int(clusterAssment[i,0]) not in groupnum:
            groupnum.append(int(clusterAssment[i,0]))
    # print(groupnum)
    # 列联表初始化K行groupnum列
    Table =[[0 for col in range(len(groupnum))] for row in range(K)]
    # 列联表获取
    for i in range(m):
        for k in range(len(groupnum)):
            if int(clusterAssment[i,0]) == groupnum[k]:
                Table[int(K*i/m)][k] += 1
    # print(Table)
    return Table,len(groupnum) #返回列联表和实际分簇的个数

#做聚类的外部度量验证，根据列联表，计算纯度、F度量（基于匹配的度量）、条件熵entropy（基于熵的度量）、Jaccard系数、Rand统计量、FM度量（成对度量）
def OuterEvaluation(contingenceTable,m,r):
    # 计算纯度，值越大越好
    Measure_sum =0
    for i in range(K):
        Measure_sum += max(contingenceTable[i])
    purity = Measure_sum/m
    # 计算F度量，值越大越好
    contingenceTable=np.mat(contingenceTable)
    F_sum=0
    for i in range(K):
        preci=max(contingenceTable.tolist()[i])/sum(contingenceTable.tolist()[i]) #先计算精度
        recalli=max(contingenceTable.tolist()[i])/sum(contingenceTable[:,i])  #再计算召回率
        Fi=2*preci*recalli/(preci+recalli)  #F度量是精度和召回率值得调和平均数
        F_sum+=Fi
    F_Measure=float(F_sum/K)
    # 计算条件熵entropy,值越小越好
    Entropy=0
    for i in range(K):
        Hi=0
        for j in range(r):
            if contingenceTable.tolist()[i][j]!=0:
                Hi+=-1*(contingenceTable.tolist()[i][j]/sum(contingenceTable.tolist()[i]))*math.log(contingenceTable.tolist()[i][j]/sum(contingenceTable.tolist()[i]),2)
        Entropy+=Hi*(sum(contingenceTable.tolist()[i])/m)
    #计算Jaccard系数(该系数只考虑了真阳性点对的比例)，Rand统计量(真阳性和真阴性的比例)、FM度量（成对精度和召回率的调和平均数）
    #以上3个值都是越大越好
    TP=0;FN=0;FP=0;TN=0
    for i in range(K):
        for j in range(r):
            if  contingenceTable.tolist()[i][j]!=0:
                TP+=0.5*contingenceTable.tolist()[i][j]*(contingenceTable.tolist()[i][j]-1)
    for i in range(K):
        FN+=0.5*sum(contingenceTable[:,i])*(sum(contingenceTable[:,i])-1)
    FN=FN-TP
    for i in range(K):
        FP+=0.5*sum(contingenceTable.tolist()[i])*(sum(contingenceTable.tolist()[i])-1)
    FP=FP-TP
    N=0.5*(m-1)*m
    TN=N-TP-FN-FP
    Jaccard=float(TP/(TP+FN+FP))
    Rand=float((TP+TN)/N)
    prec=TP/(TP+FP)
    recall=TP/(TP+FN)
    FM=math.sqrt(prec*recall)
    return purity,F_Measure,Entropy,Jaccard,Rand,FM

#计算指定两个簇的所有边的权值之和，可用欧式距离计算边的权值
def ProximityMatrix(Ci,Cj):
    m=np.shape(Ci)[0]
    n=np.shape(Cj)[0]
    W=0
    for i in range(m):
        for j in range(n):
             W+=distEuclDis(Ci[i],Cj[j])
    # print(W)
    return W

#做聚类的内部度量验证，根据簇间权值矩阵W,分为Win和Wout，计算BetaCV，计算轮廓系数SC
def InnerEvaluation(contingenceTable,r,clustering,dataset):
    m=np.shape(dataset)[0]
    contingenceTable=np.mat(contingenceTable)
    # 先计算簇内边和簇间边的数目Nin和Nout
    Nin=0;Nout=0
    for i in range(r):
        Nin+=0.5*sum(contingenceTable.tolist()[i])*(sum(contingenceTable.tolist()[i])-1)
    # print(Nin)
    for i in range(r-1):
        for j in range(i+1,r):
            Nout+=sum(contingenceTable.tolist()[i])*sum(contingenceTable.tolist()[j])
    # print(Nout)
    # 再计算簇内、簇间边的权值之和
    Win=0
    for i in range(r):
        pts=dataset[np.nonzero(clustering[:,0]==i)[0]]
        Win+=ProximityMatrix(pts,pts)
    Win=0.5*Win
    # print(Win)
    Wout=0
    for i in range(K-1):
        for j in range(i+1,K):
            ptsi=dataset[np.nonzero(clustering[:,0]==i)[0]]
            ptsj=dataset[np.nonzero(clustering[:,0]==j)[0]]
            Wout+=ProximityMatrix(ptsi,ptsj)
    # print(Wout)
    BetaCV=(Nout*Win)/(Nin*Wout)  #度量的是簇内距离均值与簇间距离均值的比值，值越小越好
    # print(BetaCV)
    # 计算轮廓系数SC，关于分簇的结合度与分离度的度量，是所有点S值的均值，值越大越好
    SC=0
    for i in range(m):
        for d in range(r):
            ptsd=dataset[np.nonzero(clustering[:,0]==d)[0]]
            if dataset[i] in ptsd:
                ClusterIndex=d  #点Xi的簇标号
        dis_Sum=0
        ptsInSameCluster=dataset[np.nonzero(clustering[:,0]==ClusterIndex)[0]]
        x=np.shape(ptsInSameCluster)[0]
        for g in range(x):
                if i!=g:
                    dis_Sum+=distEuclDis(ptsInSameCluster[g],dataset[i])
        miuIn=dis_Sum/(x-1)
        # print('1111111111',miuIn)
        minAvg=np.inf
        for q in range(r):
            dis=0
            if q!=ClusterIndex:
                ptsNotInSameCluster=dataset[np.nonzero(clustering[:,0]!=q)[0]]
                length=np.shape(ptsNotInSameCluster)[0]
                for j in range(length):
                    dis+=distEuclDis(dataset[i],ptsNotInSameCluster[j])
                dis=dis/length
                if dis<minAvg:
                    minAvg=dis
        miuOutMin=minAvg
        # print('222222222',miuOutMin)
        ssi=(miuOutMin-miuIn)/max(miuOutMin,miuIn)
        SC+=ssi
    SC=SC/m
    # print(SC)
    return BetaCV,SC

#经典K-means算法
def Kmeans(dataset,K,dist=distEuclDis,creatCent=randCentNormal):
    m=np.shape(dataset)[0] #行数
    clusterAssment=np.mat(np.zeros((m,2))) #clusterAssment保存簇分配的结果，对于每一个数据点，保存了它属于哪个簇质心的索引和平方误差距离
    centroids=creatCent(dataset,K)   #随机初始化质心
    clusterChanged=True
    iterateNum=0
    while clusterChanged:
        iterateNum+=1
        clusterChanged=False
        #------------遍历每个数据点，分配到最近的质心----------
        for i in range(m):
            minDist=np.inf #初始化为正无穷
            minIndex=-1 #保存该点到最近的质心，该质心的索引（行号）
            for j in range(K): #对于每一个数据点i，计算到质心j的距离，用disJI保存
                disJI=dist(centroids[j,:],dataset[i,:])
                if disJI<minDist:
                    minDist=disJI;minIndex=j
            if clusterAssment[i,0]!=minIndex:  #簇分配结果改变
                clusterChanged=True
                clusterAssment[i,:]=minIndex,minDist**2    #更新簇分配结果是最近质心的index,平方误差SSE（可以用来评估聚类的效果）
        #-----------计算新的均值，更新簇质心-------
        for cent in range(K):
            ptsInClust = dataset[np.nonzero(clusterAssment[:,0]==cent)[0]]  #将同一个簇的点挑出来
            centroids[cent,:]=np.mean(ptsInClust,axis=0) #将质心更新为簇中所有点的平均值，axis=0, 表示列；axis=1表示行
    return centroids,clusterAssment,iterateNum

#二分聚类算法:首先将所有点看成一个簇，对选定的簇计算总SSE，然后进行K=2的基本K-means划分，从中选择使总SSE最小的簇进行划分，并且加入这两个，直到簇数目=K
def BiKmeans(dataset,K,dist=distEuclDis):
    m=np.shape(dataset)[0]  #m为数据集的点个数
    clusterAssment=np.mat(np.zeros((m,2))) #储存簇分配结果，两列，一列是所在质心的索引，一列是平方误差SSE
    centroid0=np.mean(dataset,axis=0).tolist()[0] #计算整个数据集的质心，初始为整个数据集的平均值
    centList=[centroid0] #用列表centList保存所有质心，初始只有1个质心
    for j in range(m):     #遍历每个点，计算到初始质心的SSE（欧氏距离的平方）保存到clusterAssment的第二列
        clusterAssment[j,1]=dist(np.mat(centroid0),dataset[j,:])**2
    #-------当质心数量小于K时，不断二分簇，直到划分到有K个簇结束----------
    iterateNum=0
    while len(centList)<K :
        iterateNum+=1
        lowestSSE=np.inf
        #---------遍历当前所有簇，作二分的K-means---------
        for i in range(len(centList)):
            pInCluster=dataset[np.nonzero(clusterAssment[:,0].A==i)[0],:]#取当前簇i（质心=i）中的所有点，用Kmeans作K=2的二分
            centroidMat,splitClustAss,itera=Kmeans(pInCluster,2)
            SSESplit=sum(splitClustAss[:,1]) #划分后的总SSE=划分结果簇的SSE+未被划分的簇SSE
            SSEnotSplit=sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            #-------找到使总SSE最小的划分，将两个簇进入簇表---------
            if (SSESplit+SSEnotSplit)<lowestSSE:
                bestCentToSplit=i
                bestNewCents=centroidMat #二分后找到的使总SSE最小的质心的坐标bestNewCents
                bestClustAss=splitClustAss.copy() #二分后簇i被划分的结果
                lowestSSE=SSESplit+SSEnotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)         #被聚类的质心由新的两个质心0/1代替
        bestClustAss[np.nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit       #其中一个更新为最佳质心
        centList[bestCentToSplit]=bestNewCents[0,:].tolist()[0] #在cenList列表中添加第一个质心
        centList.append(bestNewCents[1,:].tolist()[0])  #在cenList列表中添加第二个质心
        clusterAssment[np.nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
        # showPlot(dataset,len(centList),np.mat(centList),clusterAssment)
    return np.mat(centList),clusterAssment,iterateNum

#K-means++算法，优化一般K-means的初始点的选择
def KmeansPP(dataset,K,dist=distEuclDis,creatCent=randCentpp):
    centroids=creatCent(dataset,K)
    # print('cents\n',centroids)
    m=np.shape(dataset)[0] #行数
    clusterAssment=np.mat(np.zeros((m,2)))
    clusterChanged=True
    iterateNum=0
    while clusterChanged:
        iterateNum+=1
        clusterChanged=False
        #----遍历每个数据点，分配到最近的质心---
        for i in range(m):
            minDist=np.inf #初始化为正无穷
            minIndex=-1 #保存该点到最近的质心，该质心的索引（行号）
            for j in range(K): #对于每一个数据点i，计算到质心j的距离，用disJI保存
                disJI=dist(centroids[j,:],dataset[i,:])
                if disJI<minDist:
                    minDist=disJI;minIndex=j
            if clusterAssment[i,0]!=minIndex:  #簇分配结果改变
                clusterChanged=True
                clusterAssment[i,:]=minIndex,minDist**2    #更新簇分配结果是最近质心的index,平方误差SSE（可以用来评估聚类的效果）
        #----计算新的均值向量，更新簇质心---
        for cent in range(K):
            ptsInClust = dataset[np.nonzero(clusterAssment[:,0]==cent)[0]]  #将同一个簇的点挑出来
            centroids[cent,:]=np.mean(ptsInClust,axis=0) #将质心更新为簇中所有点的平均值，axis=0, 表示列；axis=1表示行
    return centroids,clusterAssment,iterateNum

#K-medoids算法 避免噪声点影响，优化质心的更新操作
def Kmedoids(dataset,K,dist=distEuclDis,createCent=randCentNormal):
    m=np.shape(dataset)[0] #实例个数，行数
    n=np.shape(dataset)[1] #属性个数，维度，列数
    clusterAssment=np.mat(np.zeros((m,2)))
    centroids=createCent(dataset,K)
    clusterChanged=True
    iterateNum=0
    while clusterChanged :
        iterateNum+=1
        clusterChanged=False
        #----遍历每个数据点，分配到最近的质心---
        for i in range(m):
            mindist=np.inf
            minIndex=-1 #保存该点到最近的质心，该质心的索引（行号）
            for j in range(K): #对于每一个数据点i，计算到质心j的距离，用disJI保存
                disJI=dist(centroids[j,:],dataset[i,:])
                if disJI<mindist:
                    mindist=disJI;minIndex=j
            if clusterAssment[i,0]!=minIndex:  #簇分配结果改变
                clusterChanged=True
                clusterAssment[i,:]=minIndex,mindist**2
        #在每个聚簇中按照顺序依次选取点，计算该点到当前聚簇中所有点距离之和
        # 最终距离之和最小的点，则视为新的中心点
        for i in range(K):
            ptsInClust = dataset[np.nonzero(clusterAssment[:,0]==i)[0]]
            # print('簇i中所有的点\n',ptsInClust)
            minindex=np.inf
            mindist=np.inf
            for x in range(len(ptsInClust)):
                sumdist=0
                for y in range(len(ptsInClust)):
                    sumdist+=distEuclDis(ptsInClust[x],ptsInClust[y])
                    # print('点x与其它簇中点的距离之和\n',sumdist)
                if sumdist<mindist:
                    mindist=sumdist
                    minindex=x
                # print('最小质心的索引\n',minindex)
                # print('最小距离之和\n',mindist)
            centroids[i,:]=ptsInClust[minindex]
        # print('新的质心\n',centroids)
    return centroids,clusterAssment,iterateNum

def showPlot(dataset,K,centroids,clusterAssment):
    num,dim=dataset.shape
    mark=['og','ob','or','ok','op','oy','oc']
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        markIndex=int(clusterAssment[i,0])
        plt.plot(dataset[i,0],dataset[i,1],mark[markIndex])
    mark=['*r', '*y', '*g', '*k', '*b','*c']
    for i in range(K):    #对每一个质心，用不同的颜色标记
        plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=18)
    plt.show()

def showTruePlot(dataset,clusterAssment):
    num,dim=dataset.shape
    mark=['og','ob','or','oy','oc','om','sg','sb','sr','sy','sc','pg','pb','pr','py','pc']
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        markIndex=int(clusterAssment[i])
        plt.plot(dataset[i,0],dataset[i,1],mark[markIndex])
    plt.show()

def show(clustering,K,m,costTime,mycentroids,iterateNum,SSE):
    contingenceTable,groupnum=ContingenceTable(clustering,m,K)
    purity,F_measure,Entropy,Jaccard,Rand,FM=OuterEvaluation(contingenceTable,m,groupnum)
    print('--------------外部度量----------------')
    print('纯度',purity,'F度量',F_measure,'条件熵',Entropy,'Jaccard系数',Jaccard,'Rand统计量',Rand,'FM度量',FM)
    BetaCV,SC=InnerEvaluation(contingenceTable,groupnum,clustering,data)
    print('--------------内部度量----------------')
    print('BetaCV',BetaCV,'轮廓系数SC',SC)
    print('#################################################################')
    showPlot(data,K,mycentroids,clustering)

def testKmeans(data,K):
    start_time=time.time()
    mycentroids,clustering,iterateNum=Kmeans(data,K)
    SSE=sum(clustering[:,1])
    end_time=time.time()
    costTime=end_time-start_time
    m=np.shape(data)[0]
    print('Kmeans最终簇划分的质心点\n',mycentroids,'总SSE:',SSE,'所花时间:',costTime,'迭代次数:',iterateNum)
    show(clustering,K,m,costTime,mycentroids,iterateNum,SSE)

def testBiKeams(data,K):
    start_time=time.time()
    mycentroids,clustering,iterateNum=BiKmeans(data,K)
    SSE=sum(clustering[:,1])
    end_time=time.time()
    costTime=end_time-start_time
    m=np.shape(data)[0]
    print('二分Kmeans最终簇划分的质心点\n',mycentroids,'总SSE:',SSE,'所花时间:',costTime,'迭代次数:',iterateNum)
    show(clustering,K,m,costTime,mycentroids,iterateNum,SSE)

def testKmeanspp(data,K):
    start_time=time.time()
    mycentroids,clustering,iterateNum=KmeansPP(data,K)
    SSE=sum(clustering[:,1])
    end_time=time.time()
    costTime=end_time-start_time
    m=np.shape(data)[0]
    print('Kmeans++最终簇划分的质心点\n',mycentroids,'总SSE:',SSE,'所花时间:',costTime,'迭代次数:',iterateNum)
    show(clustering,K,m,costTime,mycentroids,iterateNum,SSE)

def testKmedoids(data,K):
    start_time=time.time()
    mycentroids,clustering,iterateNum=Kmedoids(data,K)
    SSE=sum(clustering[:,1])
    end_time=time.time()
    costTime=end_time-start_time
    m=np.shape(data)[0]
    print('Kmedoids最终簇划分的质心点\n',mycentroids,'总SSE:',SSE,'所花时间:',costTime,'迭代次数:',iterateNum)
    show(clustering,K,m,costTime,mycentroids,iterateNum,SSE)

if __name__ == "__main__":
    iris=datasets.load_iris()
    y=datasets.load_iris().target
    #150行2列的矩阵,属性选择第一列萼片长度和第二列萼片宽度
    data=Normalizer().fit_transform(iris.data)[:,0:2]#归一化数据
    # data=createData()
    K=3
    showTruePlot(data,y)  #画出鸢尾花数据集真实的分簇情况
    testKmeans(data,K)
    testBiKeams(data,K)
    testKmeanspp(data,K)
    testKmedoids(data,K)





