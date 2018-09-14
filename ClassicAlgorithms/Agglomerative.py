#coding:utf-8
#_author_='PYJ'

from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as hier
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import Normalizer
from sklearn import datasets
from scipy.linalg import norm
from LoadData import readFile
from sklearn.datasets import *
import time

#计算欧氏距离
def distEuclDis(vecA,vecB):
    # return sqrt(sum(power(vecA-vecB,2)))
    return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数

def createData():
    # x,y = make_blobs(n_samples=20, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2]) #生成聚类模型数据
    x,y=make_circles(n_samples=500,factor=0.2,noise=0.04)  #生成环形数据
    # x,y=make_moons(n_samples=300,noise=0.1)
    rand_data=x
    return rand_data,y

#dist_min  单链
def dist_min(dataset,Ci, Cj,clusterAssment):
    # print('Ci',Ci)
    # print('Cj',Cj)
    if Ci==Cj:
        return 0
    ptsInClusti=dataset[np.nonzero(clusterAssment[:,0]==Ci)[0]]
    ptsInClustj=dataset[np.nonzero(clusterAssment[:,0]==Cj)[0]]
    # print('ptsInClusti',ptsInClusti)
    # print('ptsInClustj',ptsInClustj)
    leni=len(ptsInClusti)
    lenj=len(ptsInClustj)
    mindist=np.inf
    for i in range(leni):
        for j in range(lenj):
            distij=distEuclDis(ptsInClusti[i],ptsInClustj[j])
            if mindist>distij:
                mindist=distij
    # print('Ci,Cj簇间最小距离',mindist)
    return mindist

#dist_max  全链
def dist_max(dataset,Ci, Cj,clusterAssment):
    # print('Ci',Ci)
    # print('Cj',Cj)
    if Ci==Cj:
        return 0
    ptsInClusti=dataset[np.nonzero(clusterAssment[:,0]==Ci)[0]]
    ptsInClustj=dataset[np.nonzero(clusterAssment[:,0]==Cj)[0]]
    # print('ptsInClusti',ptsInClusti)
    # print('ptsInClustj',ptsInClustj)
    leni=len(ptsInClusti)
    lenj=len(ptsInClustj)
    maxdist=0
    for i in range(leni):
        for j in range(lenj):
            distij=distEuclDis(ptsInClusti[i],ptsInClustj[j])
            if maxdist<distij:
                maxdist=distij
    # print('Ci,Cj簇间最大距离',maxdist)
    return maxdist


#dist_avg  均链
def dist_avg(dataset,Ci, Cj,clusterAssment):
    # print('Ci',Ci)
    # print('Cj',Cj)
    if Ci==Cj:
        return 0
    ptsInClusti=dataset[np.nonzero(clusterAssment[:,0]==Ci)[0]]
    ptsInClustj=dataset[np.nonzero(clusterAssment[:,0]==Cj)[0]]
    # print('ptsInClusti',ptsInClusti)
    # print('ptsInClustj',ptsInClustj)
    leni=len(ptsInClusti)
    lenj=len(ptsInClustj)
    maxlen=max(leni,lenj)
    sumij=0
    for i in range(leni):
        for j in range(lenj):
            distij=distEuclDis(ptsInClusti[i],ptsInClustj[j])
            sumij+=distij
    avgdist=sumij/maxlen
    # print('Ci,Cj簇间平均距离',avgdist)
    return avgdist

def FindMinDist(M):
    mindist=np.inf
    x=0;y=0
    m=np.shape(M)[0]
    for i in range(0,m,1):
        for j in range(i+1,m,1):
            if mindist>M.tolist()[i][j] and M.tolist()[i][j]!=0:
                mindist=M.tolist()[i][j]
                x=i;y=j
    return x,y,mindist

def AGNES(dataset,K,dist=dist_avg):
    #将数据集中的每个样本初始化为一个簇，并放入簇表C中。计算任意两个集合之间的距离，并存到M中
    m=np.shape(dataset)[0]
    n=np.shape(dataset)[1]
    q=m
    #-----------初始化簇分配表clusterAssment以及距离矩阵M------------
    clusterAssment = np.mat(np.zeros((m,1)))  #簇表，一列簇索引
    for i in range(m):
        clusterAssment[i]=i #初始每个数据点是一个簇
    M=np.mat(np.zeros((m,m))).tolist() #M是距离矩阵
    for i in range(m):
        for j in range(m):
            M[i][j]=distEuclDis(dataset[i],dataset[j])  #初始的距离矩阵是各个数据点之间的距离
    M=np.mat(M)
    # print(M)
    #-----------迭代地将两个最近的簇合并，直到当前聚类数目=K--------------
    while q>K:
        x,y,mindist=FindMinDist(M)
        minindex=min(x,y)
        mmindex=max(x,y)
    #------------合并两个簇--------------
        for i in range(m):
            if clusterAssment[i]==mmindex:
                clusterAssment[i]=minindex
        # print('中间产生的簇分配',clusterAssment)
    #----------更新距离矩阵M------------------
        M=np.mat(np.zeros((m,m))).tolist()
        for i in range(m):
            for j in range(m): #更新距离矩阵，用簇之间的距离计算
                Ci=clusterAssment[i]
                Cj=clusterAssment[j]
                M[i][j]=dist(dataset,Ci,Cj,clusterAssment)
        M=np.mat(M)
        q=q-1
    #     print('中间产生的距离矩阵',M)
    print("最终簇划分",clusterAssment)
    # print("最终距离矩阵",M)
    return clusterAssment,M

def showPlot(dataset,K,clusterAssment):
    plt.figure()
    num,dim=dataset.shape
    mark=['og','ob','or','ok','oy','oc','om','sg','sb','sr','sy','sc','pg','pb','pr','py','pc']
    K_list=[]
    for i in range(num):
        if clusterAssment[i] not in K_list:
            K_list.append(clusterAssment[i])
        if len(K_list) == K:
            break
    print(K_list)
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        for j in range(len(K_list)):
            if clusterAssment[i]==K_list[j]:
                plt.plot(dataset[i,0],dataset[i,1],mark[j])
    plt.show()

def PlotByDendrogram(dataset):
    disMat = hier.distance.pdist(dataset,'euclidean')
    Z = hier.linkage(disMat, method ='average',metric='euclidean')
    hier.dendrogram(Z)

def doAgenes(dataset,labels_true,K):
    clst=AgglomerativeClustering(linkage='average', n_clusters=K)
    predicted_labels=clst.fit_predict(dataset)
    labels=np.unique(labels_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbycm'
    for i,label in enumerate(labels):
        position=labels_true==label
        ax.scatter(dataset[position,0].tolist(),dataset[position,1].tolist())
    # ax.legend(loc="best",framealpha=0.5)
    # ax.set_title("data")
    plt.show()

def testAGNES(data,K):
    start_time=time.time()
    clustering,M=AGNES(data,K)
    end_time=time.time()
    costTime=end_time-start_time
    print('以下是AGNES层次聚类结果\n')
    print('--------所花时间:',costTime)
    showPlot(data,K,clustering)

if __name__ == "__main__":
    # data,answer=createData()
    data,answer=readFile("E:\毕业设计\毕业设计\DataSet\Jain.txt")
    print(data)
    print(answer)
    PlotByDendrogram(data)
    K=2
    doAgenes(data,np.array(answer),K)
    # testAGNES(data,K)










