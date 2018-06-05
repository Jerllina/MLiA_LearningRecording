# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:55:04 2018

@author: lijie
"""
import matplotlib.pyplot as plt
from numpy import *
from AdaboostForHorseDeathRatePrediction import *


def plotROC(predStrengths, classLabels):#predStrengths:the prediction strength of the classifier
    cur = (1.0,1.0) #cursor position
    ySum = 0.0 #height of the small columnar to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)#number of positive samples
    yStep = 1/float(numPosClas); 
    xStep = 1/float(len(classLabels)-numPosClas) #negative
    sortedIndicies = predStrengths.argsort()#sorted index, increase order ,therefore,plot from(1.0,1.0)
    
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    
    #loop through all the values
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep; #every sample classied to 1,y - 1*step
        else:
            delX = xStep; delY = 0;#every sample classied to the other class ,x - 1*step
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)#cursor to the next position
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve(AUC) is: ",ySum*xStep)
    
if __name__=='__main__' :
#test
    dataArr,labelArr=loadDataSet('horseColicTraining2.txt')
    classifierArr,aggClassEst=adaBoostTrainDS(dataArr,labelArr,numIt=10)

    plotROC(aggClassEst.T, labelArr)