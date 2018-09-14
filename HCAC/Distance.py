#coding:utf-8
#_author_='PYJ'
from scipy.linalg import norm
import numpy as np
import math

class Distance:
    def __init__(self,xpt,ypt):  #传入两个数据点对象，计算它们的欧氏距离
        self.xpt=xpt
        self.ypt=ypt
        self.dis=self.distEuclDis(self.xpt,self.ypt)

    #计算欧氏距离
    def distEuclDis(self,xpt,ypt):
        return math.sqrt(math.pow(xpt.xlabel - ypt.xlabel,2) + math.pow(xpt.ylabel - ypt.ylabel,2))
        # return norm(vecA-vecB)  #numpy中的linalg.norm，求范数方法，默认为2范数