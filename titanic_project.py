# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:18:48 2017

@author: Yao You
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readdata:
    train = pd.read_csv('train.csv')
    trainlean = train[['Survived', 'Pclass', 'Age', 'Sex', 'SibSp', \
                       'Parch', 'Fare','Embarked']]
    trainlean.Sex.apply(gendertonum)
    trainlean.Embarked.apply(embarknum)
    m = trainlean.shape[0]
    m = int(0.6 * m)
    trainset = trainlean[:m, :]
    
    return train

def gendertonum(g):
    if g == 'male':
        return 0.5
    else:
        return -0.5
    
def embarknum(e):
    if e == 'S':
        return 1
    else if e == 'C':
        return 2
    else if e == 'Q':
        return 3
    else:
        return -3