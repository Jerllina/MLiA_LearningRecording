# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:56:48 2018

@author: lijie
"""

from numpy import *

#model file input function 
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)
#pca
def pca(dataMat, topNfeat=9999999): #topNfeat: number of feature
    meanVals = mean((dataMat), axis=0)# calc mean
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)#calc covariance
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort 
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #last N largest number
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDataMat = meanRemoved * redEigVects #transform data into new dimensions
    reconMat = (lowDataMat * redEigVects.T) + meanVals #reconstruct data 
    return lowDataMat, reconMat

import matplotlib
import matplotlib.pyplot as plt
#plot original data & reconstructed data to make a comparison  
def plotdata(data1,data2):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(data1[:,0].flatten().A[0],data1[:,1].flatten().A[0],marker='x',s=90,c='yellow')
    ax.scatter(data2[:,0].flatten().A[0],data1[:,1].flatten().A[0],marker='o',s=50,c='red')

'''data cleaning on complex dataset'''
# replace NAN value (with mean  → do not influence the result)
def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #calc mean on values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
    
    
if __name__=='__main__' :

#test
    #dataMat=loadDataSet('testSet.txt')
    dataMat=replaceNanWithMean()
    lowDMat, reconMat=pca(dataMat,20)
    print('shape(lowDMat):\t',shape(lowDMat))
    #plotdata(dataMat,reconMat) 
    