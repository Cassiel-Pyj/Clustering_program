#coding:utf-8
#_author_='PYJ'
#coding:utf-8
#_author_='PYJ'

from sklearn.datasets import *
import numpy as np
from sklearn.decomposition import PCA
from MyAlgorithm.Plot import showTruePlot
from sklearn.preprocessing import Normalizer

def createData():
    x,y = make_blobs(n_samples=100, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2]) #生成聚类模型数据
    # x,y=make_circles(n_samples=100,factor=0.1,noise=0.1)  #生成环形数据
    # x,y=make_moons(n_samples=100,noise=0.1)
    rand_data=x
    return rand_data

#生成随机数据，4个高斯模型
def generate_data(sigma,N,mu1,mu2,mu3,mu4,alpha):
    global X                  #可观测数据集
    X=np.zeros((N, 2))       # 初始化X，N行2列。2维数据，N个样本
    X=np.matrix(X)
    global mu                 #随机初始化mu1，mu2，mu3，mu4
    mu=np.random.random((4,2))
    mu=np.matrix(mu)
    global excep              #期望第i个样本属于第j个模型的概率的期望
    excep=np.zeros((N,4))
    global alpha_             #初始化混合项系数
    alpha_=[0.25,0.25,0.25,0.25]
    for i in range(N):
        if np.random.random(1) < 0.1:  # 生成0-1之间随机数
            X[i,:]  = np.random.multivariate_normal(mu1, sigma, 1)     #用第一个高斯模型生成2维数据
        elif 0.1 <= np.random.random(1) < 0.3:
            X[i,:] = np.random.multivariate_normal(mu2, sigma, 1)      #用第二个高斯模型生成2维数据
        elif 0.3 <= np.random.random(1) < 0.6:
            X[i,:] = np.random.multivariate_normal(mu3, sigma, 1)      #用第三个高斯模型生成2维数据
        else:
            X[i,:] = np.random.multivariate_normal(mu4, sigma, 1)      #用第四个高斯模型生成2维数据
    print("初始化的mu1，mu2，mu3，mu4：\n",mu1,mu2,mu3,mu4)      #输出初始化的mu
    print("初始化的混合项系数\n",alpha)      #输出初始化的mu
    return X

def readFile(path):
    f = open(path,'rb')
    dataset=[]
    for line in f.readlines():  # 逐行进行处理
        data = line.decode().strip('\r\n')
        nums = data.split("  ")
        nums = [float(x.encode('utf-8').decode('utf-8-sig')) for x in nums]
        dataset.append([nums[0],nums[1]])
    dataset=np.mat(dataset)
    f.close()
    return dataset

def readUCI(path):
    f=open(path,'rb')
    lines=f.readlines()
    dataset=[]
    answerMark=[]
    for line in lines:
        data = line.decode().strip('\r\n')
        nums = data.split(",")
        nums = [float(x.encode('utf-8').decode('utf-8-sig')) for x in nums]
        answerMark.append(nums[0])
        dataset.append([float(x) for x in nums[1:len(nums)]])
    dataset=np.array(dataset)
    # print(dataset)
    # print(answerMark)
    return dataset,answerMark
    f.close()

def doPCA(datamat, dimension):
    print ('原有维度: ', len(datamat[0]))
    print ('开始降维:')
    pca = PCA(n_components=dimension) # 初始化PCA
    X = pca.fit_transform(datamat) # 返回降维后的数据
    print ('降维后维度: ', len(X[0]))
    # print (X)
    return X

#test
if __name__ == '__main__':
    # readFile("E:\毕业设计\毕业设计\DataSet\Aggregation.txt")
    dataset,answerMark=readUCI("E:\毕业设计\毕业设计\DataSet\Data_Wine.csv")
    # dataset=Normalizer().fit_transform(dataset)
    from sklearn.datasets import load_iris
    # data = load_iris()
    # y = data.target
    # X = data.data
    # print(X)
    # print(y)
    data=doPCA(dataset,2)
    showTruePlot(data,answerMark)

