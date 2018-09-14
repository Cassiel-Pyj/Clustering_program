#coding:utf-8
#_author_='PYJ'

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from LoadData import readFile
import scipy.sparse.linalg
from scipy.linalg import norm
from sklearn.datasets import *
import random
import math
import time
from kmeans import Kmeans
from Kernel_Kmeans import KernelKmeans

def createData():
    # x,y = make_blobs(n_samples=20, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2]) #生成聚类模型数据
    x,y=make_circles(n_samples=500,factor=0.5,noise=0.1)  #生成环形数据
    # x,y=make_moons(n_samples=20,noise=0.1)
    rand_data=x
    return rand_data

#计算欧氏距离
def distEuclDis(vecA,vecB):
    # return sqrt(sum(power(vecA-vecB,2)))
    return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数

#两个向量的核化，采用高斯核，sigma需自己取值
def Kernel(vecA,vecB):
    sigma=0.5
    kernel=math.exp((-1*norm(vecA-vecB)/(2*math.pow(sigma,2))))
    return kernel

#以使用归一割的最小化目标函数为例
def  Spectral_Clustering(dataset,K):
    m=np.shape(dataset)[0] #实例个数，行数
    #--------计算相似度矩阵A，使用高斯核计算点之间的成对相似性--------
    A=np.mat(np.zeros((m,m))).tolist()
    for i in range(m):
        for j in range(m):
            if i!=j:
                A[i][j]=distEuclDis(dataset[i],dataset[j])
    A=np.mat(A)
    print('A',A)
    #-------计算使用归一割的目标的矩阵，非对称拉普拉斯矩阵La（或对称拉普拉斯矩阵Ls）---------
    D=np.mat(np.zeros((m,m))).tolist()  #度数矩阵的倒数D
    for i in range(m):
            D[i][i]=1/np.sum(A.tolist()[i])
    D=np.mat(D)
    print(D)
    La=np.identity(m)-D*A  #等价于(1/D)*（D-A）
    print(La)
    #------计算归一化后La矩阵的K个最小特征值及对应的特征向量，形成一个m*K的特征矩阵，记为U------------
    vals,U = sp.sparse.linalg.eigs(La,K)  #获取特征值以及特征向量U
    print(U)
    centroids,clusterAssment,iterateNum=KernelKmeans(U,K)
    print(clusterAssment)
    return clusterAssment

def showPlot(dataset,K,clusterAssment):
    num,dim=dataset.shape
    mark=['og','ob','or','ok','oy','oc','om','ow',]
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        markIndex=int(clusterAssment[i,0])
        plt.plot(dataset[i,0],dataset[i,1],mark[markIndex])
    plt.show()

def testSC(data,K):
    start_time=time.time()
    clustering=Spectral_Clustering(data,K)
    end_time=time.time()
    costTime=end_time-start_time
    print('以下是谱聚类结果\n')
    print('--------所花时间:',costTime)
    showPlot(data,K,clustering)

if __name__ == "__main__":
    # data=createData()
    data,answer=readFile("E:\毕业设计\毕业设计\DataSet\Compound.txt")
    K=3
    testSC(data,K)

