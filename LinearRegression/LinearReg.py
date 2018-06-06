# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:10:42 2018

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

'''calculate w using OLS(最小二乘法)'''
def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=(xMat.T)*xMat
    if linalg.det(xTx)==0.0:
        print('this matrix cannot be inversed')
        return 
    ws=(xTx.I)*(xMat.T)*yMat
    return ws

'''plot the fitting curve'''
def fittingcurveplot(x,y):
    x=mat(x)
    y=mat(y)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x[:,1].flatten().A[0],y.T[:,0].flatten().A[0])#plot data points

    xSorted=mat(xArr) #sort x by increasing order
    xSorted.sort(0)
    yPre=xSorted*ws
    ax.plot(xSorted[:,1],yPre)
    plt.show()
    return yPre
    

if __name__=='__main__' :
#test
    xArr,yArr=loadDataSet('ex0.txt')
    ws=standRegres(xArr,yArr)
    print(ws)
    
    fittingcurveplot(xArr,yArr)
    #calc the correlation coefficient of prediction and actual number
    cc=corrcoef(yPre.T,yArr)
    
    
          