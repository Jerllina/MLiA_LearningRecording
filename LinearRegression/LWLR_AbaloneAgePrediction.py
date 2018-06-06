# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:48:39 2018

@author: lijie
"""
from LinearReg import *
from LocallyWeightedLinearReg import *

'''define square error caculation method'''
def  rssErr(yArr,yPreArr):
    eS=((yArr-yPreArr)**2).sum()
    return eS


if __name__=='__main__' :
#test
    xArr,yArr=loadDataSet('abalone.txt')
    
    #divide test set & train set
    xtrain=xArr[0:99]
    xtest=xArr[100:199]
    ytrain=yArr[0:99]
    ytest=yArr[100:199]
    
    #compare different k & standard linearReg
    yPre01=lwlrTest(xtest,xtrain,ytrain,k=0.1)
    e1=rssErr(ytest,yPre01.T)
    print('k=0.1,error is',e1)
    yPre1=lwlrTest(xtest,xtrain,ytrain,k=1)
    e2=rssErr(ytest,yPre1.T)
    print('k=1,error is',e2)
    yPre10=lwlrTest(xtest,xtrain,ytrain,k=10)
    e3=rssErr(ytest,yPre10.T)
    print('k=10,error is',e3)
    
    ws=standRegres(xtrain,ytrain)
    yPre00=mat(xtest)*ws
    e4=rssErr(ytest,yPre00.T.A)
    print('standard Regression,error is',e4)
