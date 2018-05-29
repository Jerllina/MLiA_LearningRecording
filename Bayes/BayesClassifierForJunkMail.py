# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:01:09 2018

@author: lijie
"""

from Bayes import *

'''text conversion'''

def textParse(bigString):    #dispose of strings whose lengths are less than 2 & Convert all To lower case
    import re                #import regular expression to obtain all words without notations
    #bigString = bigString.decode('utf-8')

    listOfTokens = re.split(r'\W+', bigString.decode('ISO-8859-1'))
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

'''email classifier'''

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):# text → wordlist
        wordList = textParse(open('email/spam/%d.txt' % i,'rb').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,'rb').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#construct vocabulary with wordlist we have obtained
    
    trainingSet = list(range(50)); testSet=[]        
    for i in range(10): #construct random train set &  test set(10 letters)
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet: #train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText
    
if __name__=='__main__' :
    spamTest()