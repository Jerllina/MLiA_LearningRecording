# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:30:56 2018

@author: lijie
"""

from numpy import *

'''construct text → word vector, class label(manually annotated)'''
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]  #'1' represents abusive word, '0' represents normal word  
    return postingList,classVec

'''construct vocabulary list''' # 词集模型 set-of-words model (nonredundant)
def createVocabList(dataSet):
    vocabSet = set([])  #create an empty set  (set:nonredundant list) 
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the document word vector set of the orignal vocabset  
    return list(vocabSet)

'''Check vocabulary&text'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: 
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

'''construct the naive bayes trainer '''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #use ones() to initialize
    p0Denom = 2.0; p1Denom = 2.0                        
    for i in range(numTrainDocs):#every train text,count the number of words in a class & in all
        if trainCategory[i] == 1: 
            p1Num += trainMatrix[i] 
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #calculate the prob vector of each classes
    p1Vect = log(p1Num/p1Denom)          
    p0Vect = log(p0Num/p0Denom)          
    return p0Vect,p1Vect,pAbusive

'''construct the naive bayes classifier'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

'''package all the functions above to use'''    
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    
'''construct another vocabulary list''' #词袋模型 bags-of-words model (redundant) (every time meeting word,vector+1)
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


if __name__=='__main__' :
#test   
    '''
    listOposts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOposts)
    print(myVocabList)
    myVec=setOfWords2Vec(myVocabList,listOposts[0])
    print(myVec)
    trainMat=[]
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    print(p0V)
    print(p1V)
    print(pAb)
    '''
    testingNB()

    
    
    
