# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 13:53:58 2018

@author: lijie
"""


def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):#create Candidate item set,size=1
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])                
    C1.sort()
    return list(map(frozenset, C1)) #frozenset unchangable不可改变 用户不可修改 

def scanD(D,Ck,minSupport): #
    ssCnt={}
    #Dm=list(map(set,dataSet))
    Dm=D
    for tid in Dm:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(Dm))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)# insert list at the head of the whole list
        supportData[key] = support
    return retList, supportData# support value

if __name__=='__main__' :
#test
    dataSet=loadDataSet()
    C1=createC1(dataSet)
    print(C1)
    D=list(map(set,dataSet))
    print(D)
    L1,suppData0=scanD(D,C1,0.5)
    print(L1)