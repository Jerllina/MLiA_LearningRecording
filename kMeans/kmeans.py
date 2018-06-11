# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:20:20 2018

@author: lijie
"""

from numpy import *

'''load data'''
def loadDataSet(fileName):      
    dataMat = []               
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

'''Euclidean distance calculation'''
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

'''random select k centriods inside the boundary'''
def randCent(dataSet, k):
    n = shape(dataSet)[1] # column 列数
    centroids = mat(zeros((k,n)))#initialize centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) #minimal of every column
        rangeJ = float(max(dataSet[:,j])-minJ) #range=max-min
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

'''kmeans algorithm'''
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to store data points assignment results(index,distance)
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point, assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment



if __name__=='__main__' :
#test
    myMat=mat(loadDataSet('testSet.txt'))
    myCentrs,clustAss=kMeans(myMat, 4, distMeas=distEclud, createCent=randCent)