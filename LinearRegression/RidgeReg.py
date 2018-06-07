# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:34:05 2018

@author: lijie
"""
from numpy import *
import matplotlib.pyplot as plt

'''load data'''
def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

'''calculate weights'''
def ridgeRegres(xMat,yMat,lam=0.2):#lam need to be modified
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix cannot be inversed")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

'''test a group of lam using function above'''
def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat=mat(yArr).T
    #regularize Y off y-mean
    yMean = mean(yMat,0)
    yMat = yMat - yMean     
    #regularize X
    xMeans = mean(xMat,0)   
    xVar = var(xMat,0)      #calc variance of Xi then use it to do regularization
    xMat = (xMat - xMeans)/xVar
    
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat



if __name__=='__main__' :
#test
    xArr,yArr=loadDataSet('abalone.txt')
    ws=ridgeTest(xArr,yArr)
    print(ws)
    # visualize changes of all weights with incresing log（lamb）
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ws)
    plt.show()



