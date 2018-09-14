#coding:utf-8
#_author_='PYJ'

import numpy as np
from scipy.linalg import norm

#计算欧氏距离(cut-off kernel使用)
def distEuclDis(vecA,vecB):
    # return sqrt(sum(power(vecA-vecB,2)))
    return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数

#根据参数t得到截断距离dc
def ComputeDC(dataset,t,dist=distEuclDis):
    n=np.shape(dataset)[0]
    disList=[]
    for i in range(n):
            for j in range(i+1,n): #距离矩阵，只需要取上三角矩阵
                disList.append(dist(dataset[i],dataset[j]))
    disList.sort()
    # print('预处理步骤：计算得到的点之间的距离矩阵.tolist,从小到大排序\n',disList)
    M=0.5*(n-1)*n
    index=int(np.round(M*t))
    dc=disList[index]
    print("计算得到的截断距离dc是:\n",dc)
    return dc


