# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 15:05:32 2018

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
    Dm=list(map(set,dataSet))
    for tid in Dm:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(list(Dm)))
    #print(numItems)
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)# insert list at the head of the whole list
        supportData[key] = support
    return retList, supportData# support value


def aprioriGen(Lk, k): #Lk: frequent item list #k:num of frequent items
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2:  #if first k-2 elements are equal → set union，new size=k
                retList.append(Lk[i] | Lk[j]) 
    return retList

def apriori(dataSet, minSupport = 0.7): #filter
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan Ck to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

if __name__=='__main__' :
#test
    dataSet=loadDataSet()
    C1=createC1(dataSet)
    L,suppData=apriori(dataSet)
    print(L)
    print(L[1])