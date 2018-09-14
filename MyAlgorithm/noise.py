#coding:utf-8
#_author_='PYJ'

from scipy.linalg import norm

#计算欧氏距离(cut-off kernel使用)
def distEuclDis(vecA,vecB):
    # return sqrt(sum(power(vecA-vecB,2)))
    return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数

# 噪声点检测
def doNoise(m,K,clusterAssment,w,pconfT):
    for i in range(m):
        ifiszero=[]
        for j in range(K):
            if w[i,j]<pconfT:              #当该点对于所有簇的概率都较小，小于预先设定的值，则视为噪声点
                ifiszero.append(0)
            else:
                ifiszero.append(1)
        if sum(ifiszero)==0:
            clusterAssment[i,0]=-2
    return clusterAssment