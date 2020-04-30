import numpy as np

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:21:08 2020

@author: abdallah mamdouh
"""


class LinearRegression:
    def __init__(self):
        self.m=0
        self.n=0
    def costFunction(self, X, y, Lambda=0):
        m, n = np.shape(X)
        y = np.reshape(y, (m, 1))
        hyp = X.dot(self.theta)
        diff = hyp - y
        error = np.sum(np.square(diff))
        reg = 0
        tempTheta=0
        if Lambda != 0:
            reg = Lambda * np.square(np.sum(self.theta[1:, :]))
            tempTheta = self.theta
            tempTheta[0]=0
        J = (1 / 2 * m) * (error + reg)
        grad = (1 / m) * (diff.T .dot(X)).reshape((n, 1))+(Lambda/m)*tempTheta
        return J, grad

    def train(self, X, y, alpha=0.01, epochs=10000, Lambda=0):
        self.m, self.n = np.shape(X)
        X = np.append(np.ones((self.m, 1)), X, axis=1)
        self.n+=1
        self.theta = np.zeros((self.n, 1))
        Jvec=[]
        for e in range(epochs):
            stage=epochs/100
            J, grad = self.costFunction(X, y, Lambda)
            self.theta = self.theta - alpha * grad
        return J,self.theta

    def predict(self, x):
        m = np.shape(x)[0]
        x = np.append(np.ones((m, 1)), x, axis=1)
        return  x.dot(self.theta)

