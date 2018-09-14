#coding:utf-8
#_author_='PYJ'

from matplotlib import pyplot as plt
import matplotlib
import time
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['axes.unicode_minus']=False

#画出决策图，直观地查看中心点及分配情况
def PlotDecisionGraph(ptdata):
    roList=[]
    dendistList=[]
    for p in ptdata:
        roList.append(p.ro)
        dendistList.append(p.dendist)
    plt.figure(1)
    plt.scatter(roList,dendistList)
    plt.xlabel('密度ρ')
    plt.ylabel('与最近密度更大的邻居的距离δ')
    plt.title('决策图')
    plt.show()
    time.sleep(5)
    plt.close(1)
    return roList,dendistList

def showGamma(gammaList):
    plt.figure(2)
    plt.scatter([x for x in range(len(gammaList))],gammaList)
    plt.xlabel('n')
    plt.ylabel('gamma=ro*dist')
    plt.title('选择聚类中心')
    plt.show()
    time.sleep(5)
    plt.close(2)

#画出最终簇分配结果图
def showPlot(ptdata,centList):
    mark=['og','ob','or','oy','oc','om','sg','sb','sr','sy','sc','pg','pb','pr','py','pc']
    for i in ptdata:
        if i in centList:
            plt.plot(i.attributes.tolist()[0][0],i.attributes.tolist()[0][1],'*k',markersize=18)
        elif i.clusterMark==-2:
            plt.plot(i.attributes.tolist()[0][0],i.attributes.tolist()[0][1],'+k')
        else:
            plt.plot(i.attributes.tolist()[0][0],i.attributes.tolist()[0][1],mark[i.clusterMark])
    plt.show()

#对降维后的数据进行可视化
def showPCAplot(ptdata,centList):
    mark=['og','ob','or','oy','oc','om','sg','sb','sr','sy','sc','pg','pb','pr','py','pc']
    for i in ptdata:
        if i in centList:
            plt.plot(i.attributes[0],i.attributes[1],'*k',markersize=18)
        elif i.clusterMark==-2:
            plt.plot(i.attributes[0],i.attributes[1],'+k')
        else:
            plt.plot(i.attributes[0],i.attributes[1],mark[i.clusterMark])
    plt.show()

#UCI真实数据的分簇
def showTruePlot(data,answermark):
    num,dim=data.shape
    mark=['og','ob','or','ok','oy','oc']
    for i in range(num):  #查看簇分配结果，对于每一个数据点，查看它所在的最近的质心的索引index，将在同一个簇中的点标记为同色
        markIndex=int(answermark[i])
        plt.plot(data[i,0],data[i,1],mark[markIndex])
    plt.title("真实数据的分簇")
    plt.show()
