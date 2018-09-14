#coding:utf-8
#_author_='PYJ'

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import metrics
import numpy as np
from scipy.linalg import norm
from LoadData import readFile
from sklearn.datasets import *
import random
import math
import time
from kmeans import Kmedoids

#计算欧氏距离
def distEuclDis(vecA,vecB):
    # return sqrt(sum(power(vecA-vecB,2)))
    return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数

def createData():
    # x,y = make_blobs(n_samples=150, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2]) #生成聚类模型数据
    x,y=make_circles(n_samples=300,factor=0.2,noise=0.04)  #生成环形数据
    # x,y=make_moons(n_samples=300,noise=0.05)
    rand_data=x
    return rand_data

def randCentKernel(dataset,K):
    centroids=np.mat(random.sample(dataset.tolist(),K))
    return centroids

#初始随机将所有点分成K个簇
def initClust(dataset,cent0,K,dist=distEuclDis):
    m=np.shape(dataset)[0] #实例个数，行数
    clusterAssment=np.mat(np.zeros((m,2)))
    #--------遍历所有点，分配到最近的质心,初始划分K个簇使用欧式距离--------
    clusterChanged=True
    while clusterChanged :
        clusterChanged=False
        for i in range(m):
            minDist=np.inf #初始化为正无穷
            minIndex=-1 #保存该点到最近的质心，该质心的索引（行号）
            for j in range(K): #对于每一个数据点i，计算到质心j的距离，用disJI保存
                disJI=dist(cent0[j,:],dataset[i,:])
                if disJI<minDist:
                    minDist=disJI;minIndex=j
            if clusterAssment[i,0]!=minIndex:  #簇分配结果改变
                clusterChanged=True
                SSE=0
                clusterAssment[i,:]=minIndex,SSE
    return clusterAssment

#两个向量的核化，采用高斯核，sigma需自己取值
def Kernel(vecA,vecB):
    sigma=0.5
    kernel=math.exp((-1*norm(vecA-vecB)/(2*math.pow(sigma,2))))
    return kernel

#使数据核化，输出一个核函数矩阵
def KernelMat(data):
    m=np.shape(data)[0]
    Kernelmat=np.mat(np.zeros((m,m)))
    for i in range(m):
        for j in range(m):
            Kernelmat[i,j]=Kernel(data[i],data[j])#高斯核
    return Kernelmat

#核-Kmeans
def KernelKmeans(dataset,K,createCent=randCentKernel):
    m=np.shape(dataset)[0] #实例个数，行数
    n=np.shape(dataset)[1] #属性个数，维度，列数
    centroids=createCent(dataset,K)
    clusterAssment=initClust(dataset,centroids,K) #初始将所有点随机分成K个簇
    clusterChanged=True
    iterateNum=0
    while clusterChanged :
        iterateNum+=1
        clusterChanged=False
        sqnorm=[]
        #--------计算各个簇均值的平方范数--------
        for cent in range(K):
            ptsInClust = dataset[np.nonzero(clusterAssment[:,0]==cent)[0]]  #将同一个簇的点挑出来
            ClustKernel=KernelMat(ptsInClust) #特征空间的均值平方范数即核矩阵K中数值的平均数
            sqnorm.append(np.mean(ClustKernel))
        #-------找出距离每个点最近的簇，再簇赋值（更新簇质心），先计算数据点xj与簇Ci的平均核值---------
        for i in range(m):
            mindist=np.inf
            minindex=-1
            for j in range(K):
                ptsInClust = dataset[np.nonzero(clusterAssment[:,0]==j)[0]]  #将同一个簇的点挑出来
                n=len(ptsInClust)  #簇中点的个数
                avg=0
                for a in range(n):
                    avg+=(1/n)*Kernel(dataset[i],ptsInClust[a])  #计算数据点xj与簇Ci的平均核值
                distij=sqnorm[j]-2*avg
                if distij<mindist:
                    mindist=distij
                    minindex=j
            if clusterAssment[i,0]!=minindex:  #簇分配结果改变
                clusterChanged=True
                clusterAssment[i,:]=minindex,mindist
        for cent in range(K):
            ptsInClust = dataset[np.nonzero(clusterAssment[:,0]==cent)[0]]  #将同一个簇的点挑出来
            ClustKernel=KernelMat(ptsInClust) #特征空间的均值平方范数即核矩阵K中数值的平均数
            centroids[cent,:]=np.mean(ClustKernel)
    return centroids,clusterAssment,iterateNum

def showPlot(dataset,K,centroids,clusterAssment):
    num,dim=dataset.shape
    mark=['og','ob','or','ok','oy','oc','om','ow']
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        markIndex=int(clusterAssment[i,0])
        plt.plot(dataset[i,0],dataset[i,1],mark[markIndex])
    mark=['*r', '*y', '*g', '*k', '*b','*c']
    plt.show()

def testKernelKmeans(data,K):
    start_time=time.time()
    mycentroids,clustering,iterateNum=KernelKmeans(data,K)
    SC=metrics.silhouette_score(data,clustering[:,0],metric='euclidean')
    end_time=time.time()
    costTime=end_time-start_time
    print('核Kmeans所花时间:',costTime,'迭代次数:',iterateNum)
    showPlot(data,K,mycentroids,clustering)

def testKmeans(data,K):
    start_time=time.time()
    mycentroids,clustering,iterateNum=Kmedoids(data,K)
    SC=metrics.silhouette_score(data,clustering[:,0],metric='euclidean')
    end_time=time.time()
    costTime=end_time-start_time
    print('Kmedoids所花时间:',costTime,'迭代次数:',iterateNum)
    showPlot(data,K,mycentroids,clustering)

if __name__ == "__main__":
    data=createData()
    # data,answer=readFile("E:\毕业设计\毕业设计\DataSet\G.txt")
    print(data)
    K=2
    testKernelKmeans(data,K)
    testKmeans(data,K)





