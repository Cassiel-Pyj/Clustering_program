#coding:utf-8
#_author_='PYJ'

import numpy as np

#产生簇对池，传入参数数据集个数n以及人为设定的簇对池大小sizeOfPool，以及当前在判定的两个簇标号x,y和当前的簇的距离矩阵disMat
def GeneratePool(n,sizeOfPool,x,y,disMat):
    print('当前簇的距离矩阵：\n',disMat)
    P=[]
    P.append([x,y])
    Mindist=[]
    m,n=np.shape(disMat)
    print("当前簇距离矩阵的大小",m,n)
    for i in range(0,m,1):
        for j in range(i+1,m,1):
            if (i!=j) and (((i==x and j!=y) or (j==x and i!=y)) or ((i==y and j!=x) or (j==y and i!=x))):
                Mindist.append([disMat.tolist()[i][j],i,j])
    Mindist.sort()
    print('离当前元素最小的距离的，距离和元素对:\n',Mindist)
    for t in range(sizeOfPool-1):
        if t<len(Mindist):
            P.append([int(Mindist[t][1]),int(Mindist[t][2])])
    print('簇对池：\n',P)
    return P

