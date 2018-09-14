#coding:utf-8
#_author_='PYJ'

from matplotlib import pyplot as plt
import matplotlib
import time
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['axes.unicode_minus']=False

def showGamma(gammaList):
    plt.figure(2)
    plt.scatter([x for x in range(len(gammaList))],gammaList)
    plt.xlabel('n')
    plt.ylabel('gamma=ro*dist')
    plt.title('选择聚类中心')
    plt.show()
    time.sleep(5)
    plt.close(2)

def showPlot(dataset,clusterAssment):
    num,dim=dataset.shape
    mark=['og','ob','or','oy','oc','om','sg','sb','sr','sy','sc','pg','pb','pr','py','pc']
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        markIndex=int(clusterAssment[i,0])
        if markIndex==-2:
            plt.plot(dataset[i,0],dataset[i,1],'+k')
        else:
            plt.plot(dataset[i,0],dataset[i,1],mark[markIndex])
    plt.show()

#画出最终簇分配结果图
def showPlot2(ptdata,centList):
    mark=['og','ob','or','oy','oc','om','sg','sb','sr','sy','sc','pg','pb','pr','py','pc']
    for i in ptdata:
        if i in centList:
            plt.plot(i.attributes[0],i.attributes[1],'*k',markersize=18)
            # pass
        elif i.clusterMark==-2:
            plt.plot(i.attributes[0],i.attributes[1],'+k')
        else:
            plt.plot(i.attributes[0],i.attributes[1],mark[i.clusterMark])
    plt.show()

def showTruePlot(data,answermark):
    num,dim=data.shape
    mark=['og','ob','or','ok','op','oy','oc']
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        markIndex=int(answermark[i])
        plt.plot(data[i,0],data[i,1],mark[markIndex])
    plt.title("真实数据的分簇")
    plt.show()

