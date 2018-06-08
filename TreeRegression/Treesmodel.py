# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:44:42 2018

@author: lijie
"""
from numpy import *
from TreesReg import createTree

'''load data'''
def loadDataSet(fileName):     
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #read in lines, map all elements to float()
        dataMat.append(fltLine)
    return dataMat 

#function of modelling the leaf nodes
def linearSolve(dataSet):   
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))#create a copy of data with 1 demention in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

#build leaf nodes
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

#calc square error
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yPre = X * ws
    return sum(power(Y - yPre,2))

if __name__=='__main__' :
#test
    myMat=mat(loadDataSet('exp2.txt'))
    mytree=createTree(myMat,modelLeaf,modelErr,(1,10))
    print(mytree)