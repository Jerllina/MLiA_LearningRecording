# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 18:52:12 2018

@author: lijie
"""

from apriori_FrequentItemSet import *

def generateRules(L, supportData, minConf=0.7):  #supportData: a dict  from scanD
    bigRuleList = []
    H1=[]
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    #print(H1)
    return bigRuleList  #rule list including confidence degree

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf)) #brl:checked list
            prunedH.append(conseq)
    #print(prunedH)
    return prunedH #rule list

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates , unrepetitive
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

 
if __name__=='__main__' :
#test
    dataSet=loadDataSet()
    D = map(set, dataSet)
    #print(dataSet)
    C1=createC1(dataSet)
    #print(C1)
    L,suppData=apriori(dataSet,minSupport=0.5)
    #print(L)
    #print(suppData)
    rules=generateRules(L,suppData,minConf=0.7)
    print(rules)




