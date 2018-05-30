# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:43:26 2018

@author: lijie
"""
from numpy import *
import matplotlib.pyplot as plt

'''load data'''
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt','rb')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

'''sigmoid function'''
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''grad descent function'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix format
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix format
    m,n = shape(dataMatrix)
    alpha = 0.001 #steplength
    maxCycles = 500 #iterations times
    weights = ones((n,1)) #initialize the weights
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)              #vector subtraction(real matrix - prediction matrix)
        weights = weights + alpha*dataMatrix.transpose()*error #modify the weights
    return weights

'''plot the decision boundry'''
def plotBestFit(weights):
    dataArr = array(dataMat)
    n = shape(dataArr)[0] #row number
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    x = arange(-3.0, 3.0, 0.1) #1x60
    #print(x) examine the size
    y = (-weights[0]-weights[1]*x)/weights[2] #1x60
    #print(y) examine the size
    ax.plot(x, y.transpose())
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

'''random grad descent function'''
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    dataMat=array(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize the weights
    for i in range(m):
        h = sigmoid(sum(dot(dataMat[i],weights.transpose()))) #a number
        error = classLabels[i] - h  #a number
        weights =weights+alpha*error*dataMat[i] #everytime update a weight
    return weights

'''improved random grad descent function'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):#iter times here=150,can be modified
    m,n = shape(dataMatrix)
    dataMat=array(dataMatrix)
    weights = ones(n)   #initialize 
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #alpha decreases with iteration without going to 0
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dot(dataMat[randIndex],weights)))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            del(dataIndex[randIndex])
    return weights

if __name__=='__main__' :
#test
    dataMat,labelMat=loadDataSet()
    w1=gradAscent(dataMat, labelMat)
    plotBestFit(w1)
    #print(w1)
    w2=stocGradAscent0(dataMat, labelMat)
    plotBestFit(w2)
    #print(w2)
    w3=stocGradAscent1(dataMat, labelMat, numIter=150)
    plotBestFit(w3)
    #print(w3)