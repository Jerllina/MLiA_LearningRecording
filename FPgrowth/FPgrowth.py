# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:30:26 2018

@author: lijie
"""
#class treenode: to stucture data
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None #link similar node
        self.parent = parentNode      
        self.children = {} 
        
    #calc count 
    def inc(self, numOccur):
        self.count += numOccur
    
    #disp tree as text
    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

#Construct the FP tree
def createTree(dataSet, minSup=1): 
    headerTable = {}
    #1st time traversal:count frequency
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    #filtering: delete items below minSup
    for k in list(headerTable.keys()):  
        if headerTable[k] < minSup: 
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    #print('freqItemSet: ',freqItemSet)
    
    if len(freqItemSet) == 0: 
        return None, None  #no items meet min support → end
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link 
    #print( 'headerTable: ',headerTable)
    retTree = treeNode('Null Set', 1, None) #create tree
    
    #2nd time traversal:sort
    for tranSet, count in dataSet.items():  
        localD = {}
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    return retTree, headerTable #return tree and header table

#complete fp tree
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):  
    while (nodeToTest.nodeLink != None):    
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

#create a simple dataset to test
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat #list type
#list → dict
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict #dict type

#find prefix paths
def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats #dict type

#build conditional fp tree
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]#(sort header table) 默认顺序增序
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])

        myCondTree, myHead = createTree(condPattBases, minSup)

        if myHead != None: #mine cond FP-tree
            print('conditional tree for: ',newFreqSet)
            myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

            

            
if __name__=='__main__' :
    '''
    #test1    
    rootnode=treeNode('pyramid',9,None)
    rootnode.children['eye']=treeNode('eye',13,None)
    rootnode.children['phoenix']=treeNode('phoenix',3,None)
    rootnode.disp()
    '''

    #test2
    simdat=loadSimpDat()
    initdat=createInitSet(simdat)
    myFPtree,myHeaderTab=createTree(initdat, minSup=3)
    myFPtree.disp()# 缩进是所处的树的结点深度 数字是频度计数值
    freqItems=[]
    mineTree(myFPtree, myHeaderTab, 3,set([]), freqItems)
    print(freqItems)
    
    