# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:42:51 2018

@author: lijie
"""

'''data importation'''
from numpy import *
from imp import reload
import kNN

#Convert the text record to the matrix format of Numpy
def file2matrix(filename):
    fr=open(filename)
    #Get the number of row
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    #For simplification,we make the 2nd dimension =3
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    #Resolve the data to a list
    for line in arrayOLines:
        #Intercept the 'enter' character
        line=line.strip()
        #insert the '\t' character
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

'''Digital eigenvalue data normalization'''    
def autoNorm(dataSet):    
    #参数0 表示可以从列中选取最小值，而不是当前行
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=datingMat.shape[0]
    #tile()将变量复制为输入矩阵同样大小的矩阵
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

'''data visualization/analysis'''
import matplotlib
import matplotlib.pyplot as plt
def dating_data_visualization():
    fig1=plt.figure()
    #create subplot（total row，total column，location of this picture）
    ax=fig1.add_subplot(111)
    ax.scatter(normdatingmat[:,1],normdatingmat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

'''Calculate classifer error rate'''
def datingErrorCount(normmat):
    hoRatio=0.10
    n=normmat.shape[0]
    numTestVecs=int(n*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classierfierResult=classify0(normmat[i,:],normmat[numTestVecs:n,:],datingLabels[numTestVecs:n],3)
        if classierfierResult!=datingLabels[i]:
            errorCount+=1.0
    print('the total error rate is:',(errorCount/numTestVecs))
        
'''Individual data input & prediction'''
def classifyPerson(normmat):
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input('percentage of time spent playing video games?'))
    ffMiles=float(input('frequent flier miles earned per year?'))
    iceCream=float(input('liters of icecream consumed per year?'))
    inArr=array([ffMiles,percentTats,iceCream])
    classierfierResult=classify0((inArr-minVals)/ranges,normmat,datingLabels,3)
    print('Your dating probability with this person is:',resultList[classierfierResult-1])
    
   

if __name__ == '__main__':
    datingMat,datingLabels=file2matrix('datingTestSet2.txt')
    normdatingmat,ranges,minVals=autoNorm(datingMat)
    print('normdatingMat:/t',normdatingmat)
    print('datingLabels:/t',datingLabels)
    dating_data_visualization()
    datingErrorCount(normdatingmat)
    
