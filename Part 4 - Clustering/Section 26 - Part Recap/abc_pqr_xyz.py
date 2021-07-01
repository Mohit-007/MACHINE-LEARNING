# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt # (it plot the graph)
import pandas as pd             # (it import the data)
import math
import random


# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)




rows, cols = (5,2)
means = [[0 for i in range(cols)] for j in range(rows)]

rows, cols = (200,2)
cluster1 = [[0 for i in range(cols)] for j in range(rows)]
cluster2 = [[0 for i in range(cols)] for j in range(rows)]
cluster3 = [[0 for i in range(cols)] for j in range(rows)]
cluster4 = [[0 for i in range(cols)] for j in range(rows)]
cluster5 = [[0 for i in range(cols)] for j in range(rows)]

rows, cols = (200,5)
dist = [[0 for i in range(cols)] for j in range(rows)]

for i in range(0,5):
    for j in range(0,1):
        means[i][j] = random.uniform(-2,3)

for i in range(0,5):
    for j in range(1,2):
        means[i][j] = random.uniform(-2,2)



for t in range(0,1000):
    for i in range(0,200):
         for j in range(0,5):
             dist[i][j] = math.sqrt(math.pow((X[i][0] - means[j][0]),2) + math.pow((X[i][1] - means[j][1]),2)) 
                    
        
    for i in range(0,200):
        s = dist[i].index(min(dist[i]))
        if s == 0:
            cluster1[i][0],cluster1[i][1] = X[i][0],X[i][1]
        elif s == 1:
            cluster2[i][0],cluster2[i][1] = X[i][0],X[i][1]
        elif s == 2:
            cluster3[i][0],cluster3[i][1] = X[i][0],X[i][1]
        elif s == 3:
            cluster4[i][0],cluster4[i][1] = X[i][0],X[i][1]
        elif s == 4:
            cluster5[i][0],cluster5[i][1] = X[i][0],X[i][1]
    
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0    
        
    for i in range(0,200):
        if cluster5[i][0] != 0:
            count5 = count5+1
            
    for i in range(0,200):
        if cluster4[i][0] != 0:
            count4 = count4 +1
    
    for i in range(0,200):
        if cluster3[i][0] != 0:
            count3 = count3 +1
    
    for i in range(0,200):
        if cluster2[i][0] != 0:
            count2 = count2+1
    
    for i in range(0,200):
        if cluster1[i][0] != 0:
            count1 = count1+1


    sum1 = 0
    sum2 = 0    
    for i in range(0,len(cluster1)):
        if cluster1[i][0] !=0:
           sum1 = sum1 + cluster1[i][0]
           sum2 = sum2 + cluster1[i][1]
    means[0][0] = sum1 / count1
    means[0][1] = sum2 / count1
    
    sum1 = 0
    sum2 = 0    
    for i in range(0,len(cluster2)):
        if cluster2[i][0] !=0:
            sum1 = sum1 + cluster2[i][0]
            sum2 = sum2 + cluster2[i][1]
    means[1][0] = sum1 / count2
    means[1][1] = sum2 / count2
        
    sum1 = 0
    sum2 = 0    
    for i in range(0,len(cluster3)):
        if cluster3[i][0] !=0:
            sum1 = sum1 + cluster3[i][0]
            sum2 = sum2 + cluster3[i][1]
    means[2][0] = sum1 / count3
    means[2][1] = sum2 / count3
    
    sum1 = 0
    sum2 = 0    
    for i in range(0,len(cluster4)):
        if cluster4[i][0] !=0:
            sum1 = sum1 + cluster4[i][0]
            sum2 = sum2 + cluster4[i][1]
    means[3][0] = sum1 / count4
    means[3][1] = sum2 / count4

    sum1 = 0
    sum2 = 0    
    for i in range(0,len(cluster5)):
        if cluster5[i][0] !=0:
            sum1 = sum1 + cluster5[i][0]
            sum2 = sum2 + cluster5[i][1]
    means[4][0] = sum1 / count5
    means[4][1] = sum2 / count5
    



means = sc.inverse_transform(means)
cluster5 = sc.inverse_transform(cluster5)
cluster4 = sc.inverse_transform(cluster4)
cluster3 = sc.inverse_transform(cluster3)
cluster2 = sc.inverse_transform(cluster2)
cluster1 = sc.inverse_transform(cluster1)
        