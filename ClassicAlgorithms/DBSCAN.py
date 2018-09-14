#coding:utf-8
#_author_='PYJ'

from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from LoadData import readFile
from scipy.linalg import norm
from sklearn.datasets import *
import time

def createData():
    # x,y = make_blobs(n_samples=20, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2]) #生成聚类模型数据
    # x,y=make_circles(n_samples=300,factor=0.1,noise=0.1)  #生成环形数据
    x,y=make_moons(n_samples=300,noise=0.11)
    rand_data=x
    return rand_data

#计算欧氏距离
def distEuclDis(vecA,vecB):
    # return sqrt(sum(power(vecA-vecB,2)))
    return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数

#递归地找出点x的所有的密度连通的点,并分派簇
def DensityConnected(x,k,mainPoint,clusterAssment,dataset,e):
    m=np.shape(dataset)[0]
    # print([ i for i in dataset.tolist() if distEuclDis(x,np.array(i)) <= e])
    for y in [ i for i in dataset.tolist() if distEuclDis(x,np.array(i)) <= e]:
        # print('y:',y)
        for i in range(m):
            indexIndata=0
            if dataset[i].tolist()==y:
                indexIndata=i
                break
        if clusterAssment[indexIndata]!=0:
            continue
        clusterAssment[indexIndata]=k
        # print(clusterAssment)
        if y in mainPoint:
            DensityConnected(y,k,mainPoint,clusterAssment,dataset,e)
    return 0


def DBSCAN(dataset,e,minpts):
    m=np.shape(dataset)[0]
   #-----找出所有核心点，建立核心点列表mainPoint--------
    mainPoint=[]
    for xi in dataset.tolist():
        if len([ i for i in dataset.tolist() if distEuclDis(np.array(xi),np.array(i)) <= e]) >= minpts:
            mainPoint.append(xi)
    print("核心点",mainPoint)
    #--------从每一个未分配的核心点开始，递归地找出它所有的密度连通的点，并分派给其中一个簇----------
    clusterAssment = np.mat(np.zeros((m,1)))  #簇的分配结构,是簇索引
    k=0 #分簇标识符
    for xi in mainPoint:
        # print('xi',xi)
        for i in range(m):
            indexIndata=0
            if dataset[i].tolist()==xi:
                indexIndata=i
                break
        if clusterAssment[indexIndata]!=0:
            continue
        k+=1
        clusterAssment[indexIndata]=k #进行点的簇分配
        # print(clusterAssment)
        DensityConnected(xi,k,mainPoint,clusterAssment,dataset,e)  #递归地找出它所有的密度连通的点,并分派簇
    #---------噪声点的处理--------------
    noise=[]
    noiseIndex=0
    for x in clusterAssment:
        noiseIndex+=1
        if x==0:
            noise.append(dataset[noiseIndex-1].tolist())
    # print('噪声点',noise)
    # print(clusterAssment)
    # error=clusterError(clusterAssment,m)
    return clusterAssment,noise

# def showPlot(dataset,clusterAssment,noise):
#     num,dim=dataset.shape
#     mark=['og','ob','or','ok','oy','oc','om','sg','sb','sr','sy','sc','pg','pb','pr','py','pc']
#     K_list=[]
#     for i in range(num):
#         if clusterAssment[i] not in K_list:
#             K_list.append(clusterAssment[i])
#     # print(K_list)
#     for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
#         for j in range(len(K_list)):
#             if clusterAssment[i]==K_list[j] and clusterAssment[i]!=0:
#                 plt.plot(dataset[i,0],dataset[i,1],mark[j])
#     for i in range(len(noise)):
#         plt.plot(noise[i][0],dataset[i][1],'+b')
#     plt.show()

def showPlot(dataset,clusterAssment):
    num,dim=dataset.shape
    mark=['og','ob','or','ok','oy','oc','om','sg','sb','sr','sy','sc','pg','pb','pr','py','pc']
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        if int(clusterAssment[i])!=0:
            markIndex=int(clusterAssment[i])
            plt.plot(dataset[i,0],dataset[i,1],mark[markIndex])
        else:
            plt.plot(dataset[i,0],dataset[i,1],'+b')
    plt.show()

def testDBSCAN(data,e,minpts):
    start_time=time.time()
    clustering,noise=DBSCAN(data,e,minpts)
    print("簇标签",clustering[:,0])
    end_time=time.time()
    costTime=end_time-start_time
    print('以下是DBSCAN密度聚类结果\n')
    print('--------所花时间:',costTime)
    print('噪声点',noise)
    # print('错误分簇的数据个数:',error)
    showPlot(data,clustering)

if __name__ == "__main__":
    # data = np.mat([[1, 1],[5, 6],[1, 3],[5, 5], [1, 2], [5, 7],[15, 14],[15, 15],[15, 16]])
    # data = np.mat([[1, 1], [1, 2],[15,15], [5, 5], [1,3],[15, 16],[35,36],[35,35]])
#     data = """
# 1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
# 6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
# 11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
# 16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
# 21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
# 26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
#     a = data.split(',')
#     print(a)
#     data = [(float(a[i]), float(a[i+1])) for i in range(1, len(a)-1, 3)]
#     data=np.mat(data)
    data=createData()
    testDBSCAN(data,0.2,6)