# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:09:49 2018

@author: lijie
"""

from adaboosting import *
from SimpleAdaboostClassifier import *

'''a real data classification sample'''

'''load data'''
def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat



if __name__=='__main__' :
#test
    dataArr,labelArr=loadDataSet('horseColicTraining2.txt')
    classifier=adaBoostTrainDS(dataArr,labelArr,numIt=10)
    classifierArr=classifier[0]
    #print(classifierArr)
    testArr,labelArr2=loadDataSet('horseColicTest2.txt')
    prediction=adaClassifier(testArr,classifierArr)
    errArr=mat(ones((67,1)))
    err=errArr[prediction!=mat(labelArr2).T].sum()
    errRate=err/67 #the size of testArr is 67
    print('the prediction error rate of the classifier is:',errRate)
