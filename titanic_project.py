# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:18:48 2017

@author: Yao You
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readdata(fname):
    train = pd.read_csv(fname)
    """ only use a small subset of features for now"""
    trainlean = train.loc[:, ['Survived', 'Pclass', 'Age', 'Sex', 'SibSp', \
                       'Parch', 'Fare','Embarked']]
    """ digitize string values """
    trainlean.loc[:, 'Sex'] = trainlean.loc[:, 'Sex'].apply(gendertonum)
    trainlean.loc[:, 'Embarked'] = trainlean.loc[:, 'Embarked'].apply(embarknum)
    """ drop nan """
    trainlean = trainlean.dropna(how='any')
    """ normalize the features and save the normalization parameters """
    raw_mean = trainlean.mean()
    raw_std = trainlean.std()
    for i in range(1, trainlean.shape[1]):
        trainlean.iloc[:, i] = (trainlean.iloc[:, i] - raw_mean[i]) / raw_std[i]
        
    m = trainlean.shape[0]
    m = int(0.6 * m)
    trainset = trainlean.loc[:m, :]
    vadset = trainlean.loc[m:, :]
    
    return {'train':trainset, 'validation':vadset, 'rawmean':raw_mean, \
            'rawspan':raw_std}

def gendertonum(g):
    if g == 'male':
        return 0.5
    else:
        return -0.5
    
def embarknum(e):
    if e == 'S':
        return 1
    elif e == 'C':
        return 2
    elif e == 'Q':
        return 3
    else:
        return -3

def main():
    datain = readdata('train.csv')
   

if __name__ == '__main__':
    main()     