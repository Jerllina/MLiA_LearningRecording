# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:34:08 2018

@author: lijie
"""

from TreesReg import *
'''
##预剪枝 prepruning##
#TreesReg 建造的树（已通过提前终止条件来预剪枝）
tree1={'spInd': 1, 'spVal': 0.39435, 'left': {'spInd': 1, 'spVal': 0.582002, 'left': {'spInd': 1, 'spVal': 0.797583, 'left': 3.9871632, 
    'right': 2.9836209534883724}, 'right': 1.980035071428571}, 'right': {'spInd': 1, 'spVal': 0.197834, 'left': 1.0289583666666666, 'right': -0.023838155555555553}}
# （不预剪枝的话
tree2=createTree(myMat,ops=(0,1))
#print(tree2)
#以下是结果 太长太细致 有省略 只保留了头尾 过拟合#
#{'spInd': 1, 'spVal': 0.39435, 'left': {'spInd': 1, 'spVal': 0.582002, 'left': {'spInd': 1, 'spVal': 0.797583, 'left': 
#    {'spInd': 1, 'spVal': 0.819006, 'left': {'spInd': 1, 'spVal': 0.832693, 'left': {'spInd': 1, 'spVal': 0.867298, 'left': {'spInd': 1, 'spVal': 0.872288, 'left':
#   。。。。。。。。。。。。。。
#    'right': {'spInd': 1, 'spVal': 0.013643, 'left': -0.063215, 'right': -0.067698}}, 'right': -0.217283}, 'right': -0.006338}}}}}, 'right': 0.188975}}}}}


#停止条件to1S 对误差数量级非常敏感 
tree3=createTree(myMat,ops=(10000,4))
#print(tree3)
#结果只有一个结点
#2.00369868  
'''


'''后剪枝 postpruning'''
#test if it is a tree/leaf node
def isTree(obj):
    return (type(obj).__name__=='dict')

#traverse the tree to find leaf nodes, calculate the mean of 2 leaf-nodes
def getMean(tree):
    if isTree(tree['right']): 
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): 
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

#pruning
def prune(tree, testData):
    if shape(testData)[0] == 0: 
        return getMean(tree) #if  no test data,collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if trees , split &prune
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):  
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] =  prune(tree['right'], rSet)
    #if  both leafs,  merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree
    
if __name__=='__main__' :
#test
    myMattest=mat(loadDataSet('ex2test.txt'))
    myMat2=mat(loadDataSet('ex2.txt'))
    newtree=createTree(myMat2,ops=(1,0))
    NewTree=prune(newtree,myMattest)
    print(NewTree)


