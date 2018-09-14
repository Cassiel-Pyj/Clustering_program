#coding:utf-8
#_author_='PYJ'

#coding:utf-8
#_author_='PYJ'

from scipy.sparse import identity
from scipy.linalg import norm
import math
import random
from MyAlgorithm.LoadData import *
from MyAlgorithm.Evaluation import *
from MyAlgorithm.Plot import *
from MyAlgorithm.Point import *
from MyAlgorithm.computeDC import *
import time

#初始化高斯混合分布的模型参数
def InitPara(dataset,K):
    n=np.shape(dataset)[1] #属性个数，维度，列数
    miu=np.mat(random.sample(dataset.tolist(),K)).tolist()  #初始化K个均值
    # print(miu)
    # miu=np.mat([[5,35],[30,40],[20,20],[45,15]]).tolist()
    cov=[np.mat(np.identity(n).tolist()) for x in range(K)] #初始化K个协方差矩阵为单位矩阵
    P=[1/K for x in range(K)]     #初始化K个先验概率
    return miu,cov,P

#多元高斯分布的概率密度函数f计算
def gaussProb(x,miu,cov,n):
    left=1/((math.pow(2*math.pi,n/2)*math.pow(np.linalg.det(cov),0.5)))
    right=(-0.5*(x-miu)*cov.I*(x-miu).T).tolist()
    f=left*np.exp(right)
    return f

def EMCluster(dataset,K):
     m=np.shape(dataset)[0] #实例个数，行数
     n=np.shape(dataset)[1] #属性个数，维度，列数
     miu,cov,p=InitPara(dataset,K)
     # print('初始的均值miu,cov,p',miu,cov,p)
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
         #----------M（最大化）步骤， 已知权重w，计算簇参数的极大似然估计，重新估计模型参数miu,,cov,p---------
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
         # print(w)
         #---------簇分配，将点Xi对簇Cj的贡献值大的点Xi赋给簇---------
         for i in range(m):
            clusterAssment[i,:]=np.argmax(w[i,:]), np.amax(w[i,:]) #amax返回矩阵最大值，argmax返回矩阵最大值所在下标
     return clusterAssment,iterateNum,miu,p

def testEMCluster(data,K):
    start_time=time.time()
    clustering,iterateNum,miu,p=EMCluster(data,K)
    end_time=time.time()
    costTime=end_time-start_time
    print('以下是EM高斯混合聚类结果')
    print('--------所花时间:',costTime,'迭代次数:',iterateNum,'-------')
    print('聚类出的mu',miu)
    print('聚类出的混合项系数\n',p)
    showPlot(data,clustering)
    return miu,p

if __name__ == "__main__":
    # data,answerMark=readUCI("E:\毕业设计\毕业设计\DataSet\Data_Soybean.csv")
    # data=Normalizer().fit_transform(data)
    # data=np.mat(doPCA(data,2))
    N=500        #样本数目
    K=4           #高斯模型数
    u1=[5,35]
    u2=[30,40]
    u3=[20,20]
    u4=[45,15]
    sigma=np.matrix([[30, 0], [0, 30]])               #协方差矩阵
    alpha=[0.1,0.2,0.3,0.4]         #混合项系数
    data=generate_data(sigma,N,u1,u2,u3,u4,alpha)     #生成数据

    # print(data)
    # showTruePlot(data,answerMark)
    miu,p=testEMCluster(data,4)
    # labelsPredict=[x[0] for x in clustering[:,0].tolist()]












