#coding:utf-8
#_author_='PYJ'

from PIL import Image
from scipy.cluster.vq import *
from scipy.misc import imresize
import numpy as np
from matplotlib import pyplot as plt
from kmeans import Kmeans,Kmedoids,KmeansPP,BiKmeans

#先载入图像，然后用一个 steps×steps 的窗口在图像中滑动，
# 在RGB三通道上，分别求窗口所在位置中窗口包含像素值的平均值作为特征，对这些特征利用算法进行聚类
def clusterpixels(infile, k, steps):
    im = np.array(Image.open(infile))
    dx = int(im.shape[0] / steps)
    dy = int(im.shape[1] / steps)
    # 计算每个组件的图像特征
    features = []
    for x in range(steps):
        for y in range(steps):
            R = np.mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])  #行、列、颜色通道
            G = np.mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = np.mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = np.array(features, 'f')     # 将特征值变换为数组矩阵形式
    # 聚类， k是聚类数目
    centroids, variance ,iternum= Kmeans(features, k)
    code, distance = vq(features, centroids) #进行矢量量化，使用获得的聚类标签画图
    codeim = code.reshape(steps, steps)  #给数组一个新的形状而不改变其数据
    codeim = imresize(codeim, im.shape[:2], 'nearest')  #imresize() 方法,用来指定新图像的大小
    return codeim

infile_squirrel ="squirrel.jpg"
im_squirrel = np.array(Image.open(infile_squirrel))
infile_LionKing = "LionKing.jpg"
im_LionKing = np.array(Image.open(infile_LionKing))
infile_keji ="keji.jpg"
im_keji = np.array(Image.open(infile_keji))
steps = [100,150] # image is divided in steps*steps region

#显示原图squirrel.jpg
plt.figure()
plt.subplot(331)
plt.title(u'Original Image')
plt.axis('off')
plt.imshow(im_squirrel)

# 用100*100的块对squirrel.jpg的像素进行聚类
codeim= clusterpixels(infile_squirrel, 2, steps[0])
plt.subplot(332)
plt.title(u'k=2,size=100')
#ax1.set_title('Image')
plt.axis('off')
plt.imshow(codeim)

# 用150*150的块对squirrel.jpg的像素进行聚类
codeim= clusterpixels(infile_squirrel, 2, steps[1])
ax1 = plt.subplot(333)
plt.title(u'k=2,size=150')
#ax1.set_title('Image')
plt.axis('off')
plt.imshow(codeim)

#显示原图LionKing.jpg
plt.subplot(334)
plt.title(u'Original Image')
plt.axis('off')
plt.imshow(im_LionKing)

# 用100*100的块对LionKing.jpg的像素进行聚类
codeim= clusterpixels(infile_LionKing, 3, steps[0])
plt.subplot(335)
plt.title(u'k=3,size=100')
#ax1.set_title('Image')
plt.axis('off')
plt.imshow(codeim)

# 用150*150的块对LionKing.jpg的像素进行聚类
codeim= clusterpixels(infile_LionKing, 3, steps[1])
plt.subplot(336)
plt.title(u'k=3,size=150')
plt.axis('off')
plt.imshow(codeim)

#显示原图keji.jpg
plt.subplot(337)
plt.title(u'Original Image')
plt.axis('off')
plt.imshow(im_keji)

# 用100*100的块对keji.jpg的像素进行聚类
codeim= clusterpixels(infile_keji, 4, steps[0])
plt.subplot(338)
plt.title(u'k=4,size=100')
#ax1.set_title('Image')
plt.axis('off')
plt.imshow(codeim)

# 用150*150的块对keji.jpg的像素进行聚类
codeim= clusterpixels(infile_keji, 4, steps[1])
plt.subplot(339)
plt.title(u'k=4,size=150')
plt.axis('off')
plt.imshow(codeim)

plt.show()