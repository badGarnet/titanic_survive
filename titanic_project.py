# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:18:48 2017

@author: Yao You
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readdata(fname, tp, vstart, features):
    train = pd.read_csv(fname)
    """ only use a small subset of features for now"""
    trainlean = train.loc[:, features]
    """ digitize string values """
    trainlean.loc[:, 'Sex'] = trainlean.loc[:, 'Sex'].apply(gendertonum)
    trainlean.loc[:, 'Embarked'] = trainlean.loc[:, 'Embarked'].apply(embarknum)
    """ drop nan """
    trainlean = trainlean.dropna(how='any')
  
    m = trainlean.shape[0]
    trainset = trainlean.loc[:int(tp * m), :]
    vadset = trainlean.loc[int(vstart * m):, :]
    
    return {'train':trainset, 'validation':vadset}

def normdata(a):
    raw_mean = a.mean(axis = 0)
    raw_std = a.std(axis = 0)
    out = a
    for i in range(0, a.shape[1]):
        out[:, i] = (a[:, i] - raw_mean[i]) / raw_std[i]
    return out
    
def xquad(X):
    m, nf = X.shape
    Xquad = np.zeros(shape = (m, nf * (nf + 1)))
    Xquad[:, 0 : nf] = X
    for i in range(1, nf + 1):
        Xquad[:, i * nf : (i+1) * nf] = X * np.dot(X[:, i - 1].reshape(m, 1), \
              np.ones(shape = (1, nf)))
    return Xquad    

def gendertonum(g):
    if g == 'male':
        return 0.5
    else:
        return 1
    
def embarknum(e):
    if e == 'S':
        return 1
    elif e == 'C':
        return 2
    elif e == 'Q':
        return 3
    else:
        return -3

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lfCost(theta, X, y, regpara):
    """ logistic cost function
    input: X, np.array mxn, m data sets, n features
        y, np.array mx1
        theta, np.array nx1
        regpara, float number
    cost = sum(-ylog(hx) - (1-y)log(1-hx))/2m + 2*regpara*sum(theta^2)/m
    gradient_j = sum(hx-y)*x_j + regpara*theta/m
    hx = g(X*theta')
    """
    m = X.shape[0]
    J = - (np.dot(y.T, np.log(sigmoid(np.dot(X, theta)))) + np.dot((1 - y.T), \
            np.log(1 - sigmoid(np.dot(X, theta))))) / m + 2 * regpara * np.dot( \
            theta[1:].T, theta[1:]) / m
    return J

def lfGradient(theta, X, y, regpara):
    """ logistic cost function
    input: X, np.array mxn, m data sets, n features
        y, np.array mx1
        theta, np.array nx1
        regpara, float number
    cost = sum(-ylog(hx) - (1-y)log(1-hx))/2m + 2*regpara*sum(theta^2)/m
    gradient_j = sum(hx-y)*x_j + regpara*theta/m
    hx = g(X*theta')
    """
    m = X.shape[0]
    grad = np.dot(X.T, sigmoid(np.dot(X, theta)) - y) / m
    grad[1:] = grad[1:] + regpara * theta[1:] / m
    return grad

def lfPredict(theta, X):
    yp = sigmoid(np.dot(X, theta))
    yp[yp > 0.5] = 1
    yp[yp <=0.5] = 0
    return yp

def main():
    datain = readdata('train.csv')
   

if __name__ == '__main__':
    main()     