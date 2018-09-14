#coding:utf-8
#_author_='PYJ'

from sklearn.datasets import *
import numpy as np
from sklearn.decomposition import PCA

def createData():
    x,y = make_blobs(n_samples=100, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2]) #生成聚类模型数据
    # x,y=make_circles(n_samples=100,factor=0.1,noise=0.1)  #生成环形数据
    # x,y=make_moons(n_samples=100,noise=0.1)
    rand_data=x
    return rand_data

def readFile(path):
    f = open(path,'rb')
    answerList=[];i=0
    marklist=[]
    dataset=[]
    for line in f.readlines():  # 逐行进行处理
        data = line.decode().strip('\r\n')
        nums = data.split("\t")
        nums = [float(x.encode('utf-8').decode('utf-8-sig')) for x in nums]
        dataset.append([nums[0],nums[1]])
        if nums[2] not in marklist:
            marklist.append(nums[2])
            answerList.append([[nums[0],nums[1]]])
        else:
            index=np.inf
            for i in range(len(marklist)):
                if nums[2]==marklist[i]:
                    index=i
                    break
            answerList[index].append([nums[0],nums[1]])
    dataset=np.mat(dataset)
    # print("dataset",dataset)
    # print("answerList",answerList)
    # print("答案列表长度",len(answerList))
    f.close()
    return dataset,answerList

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
    # print ('原有维度: ', len(datamat[0]))
    print ('开始降维:')
    pca = PCA(n_components=dimension) # 初始化PCA
    X = pca.fit_transform(datamat) # 返回降维后的数据
    print ('降维后维度: ', len(X[0]))
    # print (X)
    return X

#test
if __name__ == '__main__':
    readFile("E:\毕业设计\毕业设计\DataSet\Aggregation.txt")