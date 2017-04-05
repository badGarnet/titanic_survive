# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:18:48 2017

@author: Yao You
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scop

def readdata(fname, tp, vstart, features):
    train = pd.read_csv(fname)
    """ only use a small subset of features for now"""
    trainlean = train.loc[:, features]
    """ digitize string values """
    trainlean.loc[:, 'Sex'] = trainlean.loc[:, 'Sex'].apply(gendertonum)
    trainlean.loc[:, 'Embarked'] = trainlean.loc[:, 'Embarked'].apply(embarknum)
    
    """ extract titles and map titles to age for those missing age in db
    tmap = {' Mr':0, ' Mrs':1, ' Miss':2, ' Mme':1, ' Master':3}
    title = trainlean.Name.apply(name2title).map(tmap).fillna(4).astype(int)
    medage = trainlean.groupby(title)['Age'].median()
    trainlean.loc[trainlean.Age.isnull(), 'Age'] = \
                  title[trainlean['Age'].isnull()].map(medage)
    trainlean = trainlean.assign(title = title.values)
    del trainlean['Nmae']
    """
    
    """ drop nan
    trainlean = trainlean.dropna(how='any')
    """   
  
    m = trainlean.shape[0]
    trainset = trainlean.loc[:int(tp * m), :]
    vadset = trainlean.loc[int(vstart * m):, :]
    
    return {'train':trainset, 'validation':vadset}

def regerrors(regpara, X, y, Xvad, yvad):
    """ run the model for a set of regularization parameters and return the
    training and validaton set errors """
    trainerror = np.zeros(shape = (regpara.size, 1))
    vaderror = np.zeros(shape = (regpara.size, 1))
    theta0 = np.zeros(shape = (X.shape[0], 1))

    for i in range(0, regpara.size):
        theta = scop.fmin_l_bfgs_b(lfCost, theta0, lfGradient, (X, y, regpara[i]))[0]
        trainerror[i] = lfCost(theta, X, y, regpara[i])
        vaderror[i] = lfCost(theta, Xvad, yvad, regpara[i])
    plt.semilogx(regpara, trainerror, '-b', regpara, vaderror,'-r')
    plt.xlabel('regularization parameter value')
    plt.ylabel('error')
    plt.legend(['training','validation'])
    plt.show()
    
    return {'trainerror': trainerror, 'vaderror': vaderror}

def polyx(X, order):
    m, nf = X.shape
    Xp = np.zeros(shape = (m, nf * order))
    Xp[:, 0 : nf] = np.copy(X)
    for i in range (1, order):
        Xp[:, i * nf:(i + 1) * nf] = X ** (i + 1)
    return Xp

def name2title(name):
    return name.split(',')[1].split('.')[0]

def getacu(theta, X, y):
    y_predict = sigmoid(np.dot(X, theta))
    y_predict[y_predict > 0.5] = 1
    y_predict[y_predict <= 0.5] = 0
    return (y_predict == y).sum()*1.0/y.size
    
def normdata(a):
    raw_mean = a.mean(axis = 0)
    raw_std = a.std(axis = 0)
    out = np.copy(a)
    for i in range(0, a.shape[1]):
        if raw_std[i] == 0:
            out[:, i] = (a[:, i] + 1.0) / (raw_mean[i] + 1.0)
        else:
            out[:, i] = (a[:, i] - raw_mean[i]) / raw_std[i]
    return out
    
def xquad(X):
    m, nf = X.shape
    Xquad = np.zeros(shape = (m, nf + nf * (nf + 1) / 2))
    Xquad[:, 0 : nf] = np.copy(X)
    quadind = np.insert(np.arange(nf,0,-1), 0, nf).cumsum()
    for i in range(1, nf+1):
        Xquad[:, quadind[i-1] : quadind[i]] = X[:, i - 1 : nf] * \
              np.dot(X[:, i - 1].reshape(m, 1), \
              np.float64(np.ones(shape = (1, nf + 1 - i))))
    return Xquad    

def gendertonum(g):
    if g == 'male':
        return -1
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

def lrCost(theta, X, y):
    """linear regression cost function """
    m = X.shape[0]
    return np.dot((np.dot(X, theta) - y).T, np.dot(X, theta) - y) / 2.0 / m

def lrGradient(theta, X, y):
    """ linear regression gradient """
    return np.dot(X.T, np.dot(X, theta) - y) / X.shape[0]

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