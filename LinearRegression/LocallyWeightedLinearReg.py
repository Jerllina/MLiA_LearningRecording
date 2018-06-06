# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:30:56 2018

@author: lijie
"""
from numpy import *

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

'''locally weighted linear regression'''
def lwlr(testPoint,xArr,yArr,k):#k should be given 
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))#initialize w
    for j in range(m):                     
        diffMat = testPoint - xMat[j,:]     
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("this matrix cannot be inversed")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    pointPre=testPoint * ws
    return pointPre

'''use lwlr above to loop on a dataset'''
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr 
    m = shape(testArr)[0]
    yPre = zeros(m)
    for i in range(m):
        yPre[i] = lwlr(testArr[i],xArr,yArr,k)
    return yPre

'''plot the fitting curve'''
def fittingcurveplot(x,y):
    x=mat(x)
    y=mat(y)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x[:,1].flatten().A[0],y.T[:,0].flatten().A[0],s=2,c='orange')#plot data points

    xSorted=mat(xArr) #sort x by increasing order
    xSorted.sort(0)
    yPre=xSorted*ws
    ax.plot(xSorted[:,1],yPre)
    plt.show()
    return yPre

if __name__=='__main__' :
#test
    xArr,yArr=loadDataSet('ex0.txt')
    #yPre1=lwlrTest(xArr[0],xArr,yArr,k=1.0)#estimate a point
    yPre=lwlrTest(xArr,xArr,yArr,k=0.003)#a dataset
    
    fittingcurveplot(xArr,yArr)
    #calc the correlation coefficient of prediction and actual number
    cc=corrcoef(yPre.T,yArr)



