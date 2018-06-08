# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:22:57 2018

@author: lijie
"""
from numpy import *

# create tree node
class treeNode():
    def __init__(self,feat,val,right,left):
        featureToSplitOn=feat
        valueOfSplit=val
        rightBranch=right
        leftBranch=left

'''load data'''
def loadDataSet(fileName):     
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #read in lines, map all elements to float()
        dataMat.append(fltLine)
    return dataMat   

'''binary split for 1 feature'''
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1  
#create leaves
def regLeaf(dataSet):       #return the value used for each leaf
    return mean(dataSet[:,-1]) #mean value

#error calculation
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0] #Mean variance function & sum 


'''tree construction'''
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):# dataSet → NumPy Mat ,array filtering
    dataSet=mat(dataSet)                                        #leafType:function to create leaves
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#errType: error calculating function chosen
    if feat == None: 
        return val 
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops) #递归 调用自己
    return retTree    


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0] #Allowed error reduction
    tolN = ops[1] #least data number to be splited
    #if only 1 node , end
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the error is less than a threshold , end
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue

if __name__=='__main__' :
#test
    myMat=mat(loadDataSet('ex0.txt'))
    mytree=createTree(myMat)
    print(mytree)
    