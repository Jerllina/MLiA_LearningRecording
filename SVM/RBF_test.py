# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 22:22:31 2018

@author: lijie
"""
from numpy import *
from completed_SMO import *

'''load data'''
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])#alpha1,alpha2
        labelMat.append(float(lineArr[2]))#label
    return dataMat,labelMat


'''package all the functions'''
def testRbf(k1):    #k1 : Velocity parameter of rbf
    #train
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0] #get matrix of only support vectors
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print( "there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    # key part to do classification & calculate the error rate
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print( "the training error rate is: %f" % (float(errorCount)/m))
    #test
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print ("the test error rate is: %f" % (float(errorCount)/m) )  
    
if __name__=='__main__' :
#test
    testRbf(k1=1.3)
'''    
    #the result:
        iteration number: 6
        there are 26 Support Vectors
        the training error rate is: 0.090000
        the test error rate is: 0.180000
'''
    