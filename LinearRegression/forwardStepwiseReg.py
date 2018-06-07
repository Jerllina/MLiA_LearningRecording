# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:50:20 2018

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

'''regularization to mean=0,variance=1'''
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

'''stepwise algorithm'''
def stageWise(xArr,yArr,eps,numIt):#eps:step length  numIt:iteration times
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy() #2 copies to use
    for i in range(numIt):
        print(ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssErr(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

'''define square error caculation method'''
def  rssErr(yArr,yPreArr):
    eS=((yArr-yPreArr)**2).sum()
    return eS

if __name__=='__main__' :
#test
    xArr,yArr=loadDataSet('abalone.txt')
    ws=stageWise(xArr,yArr,eps=0.001,numIt=1000)
    print(ws)
    # visualize changes of all weights with incresing log（lamb）
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ws)
    plt.show()