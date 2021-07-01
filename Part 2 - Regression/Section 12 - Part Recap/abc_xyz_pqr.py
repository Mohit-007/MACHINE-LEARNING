# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, -1:].values
"""

"""
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4:5].values
"""

"""
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
"""


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# dummy variable trap 
X = X[:, 1:]

# train test split
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
'''

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)

"""

# simple linear regression

m = 0
c = 0
a = 0.01
n = int(len(X_train))

for i in range(0,1000):
        y_pred = m*X_train + c
        error = (Y_train - y_pred)
        t0 = (-2/n)*sum(Y_train - y_pred)
        t1 = (-2/n)*sum(X_train*(Y_train - y_pred))
        m = m - a*t1
        c = c - a*t0

"""        

"""        

# multiple linear regression
        
a = 0.01
row = int(len(X_train))
col = int(len(X_train[0]))

X_train = np.append(X_train, np.ones((row,1)), axis = 1)
p = np.ones((col+1,1))
error_matrix = np.zeros((col+1,1))

for i in range(0,500):
        y_pred = X_train.dot(p)        
        error = y_train - y_pred
        for j in range(0,col+1):
            error_matrix[j] = (-2/n)*sum(error*X_train[:, j:j+1])
        p = p - a*error_matrix

row = int(len(X_test))
col = int(len(X_test[0]))
X_test = np.append(X_test, np.ones((row,1)), axis = 1)
y_pred = X_test.dot(p)

"""

"""

# polynomial linear regression 

a = 0.01
n = int(len(X))

X_1 = X
X_2 = X*X
X_3 = X*X*X
X_4 = X*X*X*X
 
X = np.append(X, np.ones((n,1)), axis = 1)

X[: , 0:1] = X_4
X[: , 1:2] = X_3
X[: , 2:3] = X_2
X[: , 3:4] = X_1


m = int(len(X[0]))

X = np.append(X, np.ones((n,1)), axis = 1)
error_matrix = np.zeros((m,1))
p = np.ones((m,1))

for i in range(0,500):
        y_pred = X.dot(p)        
        error = y - y_pred
        for j in range(0,m):
            error_matrix[j] = (-2/n)*sum(error*X[:, j:j+1])
        p = p - a*error_matrix

y_pred = X.dot(p)
y_pred = sc_y.inverse_transform(y_pred)
y = sc_y.inverse_transform(y)

"""


##################################################################################       