# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:11:15 2019

@author: RAMESHWAR LAL JI
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values 
Y = dataset.iloc[:, -1:].values



from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


m1 = 0
m2 = 0
c = 0
a = 0.01
n = int(len(X_train))



for i in range(0,1000):
        mx = m1*X_train[: , 2:3] + m2*X_train[: , 3:4] + c
        y_pred = 1/(1 + np.exp(-mx));
        "error = (Y_train - y_pred)"
        t0 = (-2/n)*sum(Y_train - y_pred)
        t1 = (-2/n)*sum(X_train[: , 2:3]*(Y_train - y_pred))
        t2 = (-2/n)*sum(X_train[: , 3:4]*(Y_train - y_pred))
       
        m1 = m1 - a*t1
        m2 = m2 - a*t2
        c = c - a*t0
        
        