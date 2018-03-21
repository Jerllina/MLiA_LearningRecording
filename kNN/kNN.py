# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:00:26 2018

@author: lijie
"""
from numpy import *
import operator

def createDataSet():
    #创建数据集和标签
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    #a**b  返回a的b次幂
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

    
    
    
    


#test:
if __name__=='__main__' :
    group,labels=createDataSet()
    Label=classify0([0,0],group,labels,3)
    print('The input belongs to:', Label)
    
    
    
