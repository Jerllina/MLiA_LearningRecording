# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:22:00 2018

@author: lijie
"""
from numpy import *
from adaboosting import *

'''simple classification test'''
def adaClassifier(dataToClass,classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        #stumpClassify(dataMatrix,dimen,threshVal,threshIneq)
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)


if __name__=='__main__' :
#test
    dataArr,labelArr=testData()
    classifier=adaBoostTrainDS(dataArr,labelArr,numIt=30)
    classifierArr=classifier[0]
    adaClassifier([0,0],classifierArr)
 

