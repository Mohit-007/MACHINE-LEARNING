# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:22:58 2019

@author: RAMESHWAR LAL JI
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)

sc = StandardScaler()
X_test = sc.fit_transform(X_test)



a = 0.05
w = np.ones((np.shape(X_train)[1],1))  
n = int(len(X_train))

for i in range(0,750):
        h = X_train.dot(w)        
        g = 1.0 / (1 + np.exp(-h))
        w = w - a*((X_train.transpose()).dot(g - Y_train))
        
 
h_p = X_test.dot(w)
 
    
for i in range(0,133):
        if -h_p[i][0] > 0 :
            y_pred[i][0] = 1
        else:
            y_pred[i][0] = 0
    