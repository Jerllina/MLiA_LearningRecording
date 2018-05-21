# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:39:18 2018

@author: lijie
"""

from math import log
import operator

'''Calculate shannon entropy'''
def calcShannonEnt(dataSet):
    #Calculate total number of Entries
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        #Count label types
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        #Count the number of every label type of entries
        labelCounts[currentLabel]+=1
    #higher shannonEnt → more types of data
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

'''Partition data set'''
#dataSet:data to be divided
#axis:feature to divide with
#value:feature to return
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            #extend:Union the data in the same dimension.
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

''''''
def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
    
'''find the best split way'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

'''Recursion to construct a decision tree'''
#get the most likely class  
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys:
            classCount[vote]=0
            classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
#create desicion tree
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #stop splitting the dataset when the classlist are the same
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
              (dataSet,bestFeat,value),subLabels)
    return myTree

        

if __name__=='__main__' :
   myDat,labels=createDataSet()
   a=splitDataSet(myDat,0,1)
   b=splitDataSet(myDat,0,0)
   c=chooseBestFeatureToSplit(myDat)
   t=createTree(myDat,labels)
   print(t)
   
   
           
            
            
            
            
            
            
            
            
            
            
            
            
        