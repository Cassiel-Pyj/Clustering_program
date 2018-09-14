#coding:utf-8
#_author_='PYJ'
from sklearn.metrics import  silhouette_score,jaccard_similarity_score,fowlkes_mallows_score,adjusted_mutual_info_score,accuracy_score

#内部度量轮廓系数SC，值越大越好
def doSilhouette(data, clusterlabel):
    print ('计算轮廓系数，值越大越好:')
    silhouette_avg = silhouette_score(data, clusterlabel) # 平均轮廓系数
    # sample_silhouette_values = silhouette_samples(X, y) # 每个点的轮廓系数
    print(silhouette_avg)
    return silhouette_avg

#外部度量Jaccard系数，值越大越好
def doJaccard(true,pred):
    print ('计算Jaccard系数，值越大越好:')
    ja=jaccard_similarity_score(true,pred)
    print(ja)
    return ja

#外部度量，AMI指标,值越小越好
def doinfo(labelstrue,labelspred):
    print ('计算AMI指标，值越小越好:')
    ac=adjusted_mutual_info_score(labelstrue,labelspred)
    print(ac)
    return ac

#外部度量FMI，值越小越好
def doFmeasure(labelstrue,labelspred):
    print ('计算FMI指标,值越小越好:')
    ac=fowlkes_mallows_score(labelstrue,labelspred)
    print(ac)
    return ac

#准确率
def doaccuracy(labelstrue,labelspred):
    print ('计算准确率,值越大越好:')
    ac=accuracy_score(labelstrue,labelspred)
    print(ac)
    return ac
