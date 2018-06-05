# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:17:46 2018

@author: lijie
"""
from numpy import * 

'''a simple test dataset'''
def testData():
    dataMat=mat([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat,classLabels

'''classify data'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#do classification 
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0 #Array slice
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

'''find decision node,build the decision stump'''
'''simplied decision tree,weighted classifier'''
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf #initialize error sum  to +∞
    for i in range(n):#1st loop :over all dimensions
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#2nd loop:over all range in current dimension
            for inequal in ['lt', 'gt']: #3rd loop:go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0 #if prediction is right ,slice=0
                weightedError = D.T*errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

'''use weighted classifier above to complete the adaboosting algorithm'''
def adaBoostTrainDS(dataArr,classLabels,numIt=40): #numIT:iteration time
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #initialize D 
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#max(error,1e-16),ensure the division when error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)     #store Stump Params in Array
        #print ("classEst: ",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst #record estimated value
        #print ("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))#slice
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: 
            break
    return weakClassArr,aggClassEst

if __name__=='__main__' :
#test
    dataArr,classLabels=testData()
    D=mat(ones((5,1))/5)
    #bestStump,minError,bestClasEst=buildStump(dataArr,classLabels,D)
    weakClassArr,aggClassEst=adaBoostTrainDS(dataArr,classLabels,numIt=40)
    print(bestStump)
    print(weakClassArr)