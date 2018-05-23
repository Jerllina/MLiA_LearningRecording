# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:35:31 2018

@author: lijie
"""
import trees
import treePlotter

def classify(inputTree,featLabels,testVec):
    firstSide=list(inputTree.keys())
    firstStr=firstSide[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

if __name__=='__main__' :
    #test
    tree=treePlotter.retrieveTree(0)
    dataset,labels=trees.createDataSet()
    a=classify(tree,labels,[1,0])
    print(a)