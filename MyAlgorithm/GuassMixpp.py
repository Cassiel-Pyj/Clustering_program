#coding:utf-8
#_author_='PYJ'

from EMcluster import gaussProb
import numpy as np
import random
import math
from MyAlgorithm.LoadData import *
from scipy.linalg import norm
from Plot import *
from noise import doNoise
from searchCent import *
from EMcluster import *

#创建K个随机miu的集合（K-means++的方法）
def randmiu(dataset,K):
    m=np.shape(dataset)[0] #数据集行的数量
    n=np.shape(dataset)[1] #数据集列的数量
    centriods=[]
    cent0=[]
    #第一个质心随机取
    for j in range(n):
        minJ=np.min(dataset[:,j])  #求数据集每一列的最小值、最大值
        maxJ=np.max(dataset[:,j])
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
        centriods.append(dataset[x].tolist()[0])
    return np.mat(centriods)

#初始化高斯混合分布的模型参数cov,P
def InitParapp(dataset,K):
    n=np.shape(dataset)[1] #属性个数，维度，列数
    cov=[np.mat(np.identity(n).tolist()) for x in range(K)] #初始化K个协方差矩阵为单位矩阵
    P=[1/K for x in range(K)]     #初始化K个先验概率
    return cov,P

# 改进的高斯混合聚类算法
def GaussMixpp(dataset,K):
    m=np.shape(dataset)[0]
    n=np.shape(dataset)[1] #属性个数，维度，列数
    #初始化高斯混合分布的模型参数cov,P
    cov,p=InitParapp(dataset,K)
    # print('初始的均值cov,p',cov,p)
    #初始化Miu均值
    # miu=randmiu(dataset,K).tolist()  #Kmeans++那种取法
    # print(miu)
    t=0.9;pconfT=0.5
    miu,dc=beginSearch(dataset,K,t)    #快速密度聚类的取法,初始化高斯混合分布的模型参数Mu
    # print('初始的均值miu',miu)
    w=np.mat(np.zeros((m,K))) #簇的后验概率，权重W
    clusterAssment = np.mat(np.zeros((m, 2)))
    e=0.1
    iterateNum=0
    clusterChanged=True
    while clusterChanged:
         clusterChanged=False
         iterateNum+=1
         oldmiu=np.copy(miu)
         #-----------E（期望）步骤，根据当前参数值，计算每个样本属于每个高斯成分的后验概率(权重)----------------
         for j in range(m):
             sumFenzi=0
             for a in range(K):
                 sumFenzi+=p[a]*gaussProb(dataset[j,:],miu[a],cov[a],n)
             for i in range(K):
                 w[j,i]=p[i]*gaussProb(dataset[j,:],miu[i],cov[i],n)/sumFenzi  #簇Ci的后验概率，点xj对簇Ci的贡献
         sumW=np.sum(w,axis=0) #按列相加
         #----------M（最大化）步骤,已知权重w，计算簇参数的极大似然估计，重新估计模型参数miu,,cov,p---------
         for i in range(K):
             miu[i]=np.mat(np.zeros((1,n)))
             cov[i]=np.mat(np.zeros((n,n)))
             #重新估计均值miu
             for j in range(m):
                 miu[i]+=w[j,i]*dataset[j,:]
             miu[i]/=sumW[0,i]
             #重新估计协方差矩阵cov
             for j in range(m):
                 cov[i]+=w[j,i]*(dataset[j,:]-miu[i]).T*(dataset[j,:]-miu[i])
             cov[i]/=sumW[0,i]
             #重新估计先验概率P
             p[i]=sumW[0,i]/m
         #---------判断收敛------------------
         condition=0
         for i in range(K):
             # print(miu[i],oldmiu[i])
             condition+=norm(miu[i]-oldmiu[i])
         # print(condition)
         if condition>e:
             clusterChanged=True
         # print('每个点对每个簇的权重矩阵\n',w)
         #---------簇分配，将点Xi对簇Cj的贡献值大的点Xi赋给簇--------
         for i in range(m):
             clusterAssment[i,:]=np.argmax(w[i,:]), np.amax(w[i,:]) #amax返回矩阵最大值，argmax返回矩阵最大值所在下标
         clusterAssment=doNoise(m,K,clusterAssment,w,pconfT)
    return clusterAssment,iterateNum,miu,p

def testGaussMixpp(data,K):
    start_time=time.time()
    clustering,iterateNum,miu,p=GaussMixpp(data,K)
    end_time=time.time()
    costTime=end_time-start_time
    print('以下是EM高斯混合++聚类结果')
    print('--------所花时间:',costTime,'迭代次数:',iterateNum,'-------')
    print('聚类出的mu',miu)
    print('聚类出的混合项系数',p)
    showPlot(data,clustering)
    return miu,p


if __name__ == "__main__":
    data=readFile("E:\毕业设计\毕业设计\DataSet\GuassMixpp.txt")

    # N=500        #样本数
    K=4           #高斯模型数
    # u1=[5,35]
    # u2=[30,40]
    # u3=[20,20]
    # u4=[45,15]
    # sigma=np.matrix([[30, 0], [0, 30]])               #协方差矩阵
    # alpha=[0.2,0.3,0.4,0.5]         #混合项系数
    # data=generate_data(sigma,N,u1,u2,u3,u4,alpha)     #生成数据
    # print(data)
    miu1,p1=testEMCluster(data,K)
    miu2,p2=testGaussMixpp(data,K)






