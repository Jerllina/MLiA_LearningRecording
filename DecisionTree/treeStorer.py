# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:55:47 2018

@author: lijie
"""

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

