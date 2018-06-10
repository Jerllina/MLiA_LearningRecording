# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:44:32 2018

@author: lijie
"""
from treePruning import *
from Treesmodel import *


#predict reg leaf node
def regTreeEval(model, inDat):
    return float(model)

#predict model leaf node
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))#incease a row of 1 第一列是1
    X[:,1:n+1]=inDat 
    return float(X*model)

#traverse the tree,calc the leaf node
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): 
        return modelEval(tree, inData)
    if inData[tree['spInd']]> tree['spVal']:
        if isTree(tree['left']): 
            return treeForeCast(tree['left'], inData, modelEval)
        else: 
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): 
            return treeForeCast(tree['right'], inData, modelEval)
        else: 
            return modelEval(tree['right'], inData)
        
# use the function above many times 
def createForeCast(tree, testData, modelEval=regTreeEval):
   m=len(testData)
   yPre = mat(zeros((m,1)))
   for i in range(m):
       yPre[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
   return yPre       

if __name__=='__main__' :
#test
    trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    mytree=createTree(trainMat,modelLeaf,modelErr,(1,20))
    print(mytree)   
    yPre=createForeCast(mytree, testMat[:,0], modelEval=modelTreeEval)
    cc=corrcoef(yPre,testMat[:,1],rowvar=0)[0,1] 
    print(cc) #closer to 1,better
