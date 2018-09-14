#coding:utf-8
#_author_='PYJ'

from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn import datasets
from matplotlib import pyplot as plt
import main

def showPlot(resultClusterAssement):
    num,dim=main.dataset.shape
    mark=['og','ob','or','ok','oy','oc','om','ow']
    K_list=[]
    for i in range(num):
        if resultClusterAssement[i] not in K_list:
            K_list.append(resultClusterAssement[i])
        if len(K_list) == main.K:
            break
    print(K_list)
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        for j in range(len(K_list)):
            if resultClusterAssement[i]==K_list[j]:
                plt.plot(main.dataset[i,0],main.dataset[i,1],mark[j])
    plt.show()




