"""

note : simple linear regression model
-> y = a + b*x
-> y = dependent variable && x = independent variable
-> a = slope && b = constant

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
 

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


# importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, 1].values

"""
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
"""


# splitting the dataset
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


"""
# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""



# fitting simple linear regression to the training set

"""

note : it will draw a line that best fits the graph (dataset point)

-> import linear regression class by linear_model of sklearn library
-> make object regressor of the class and call linear regression method
-> fit the regressor object in the independent and dependent training set 

"""

# from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
 
# predicting the test set results

"""

note : it will predict the value of dependent variable for test set values of independent variable 

-> call predict method via regressor object (give independent test set data)
-> it will return the vector of independent varaiable 
-> it will give (dependent value of best fitting line on independent variable) 

"""

Y_pred = regressor.predict(X_test)

# visualising the training set results

"""

note : it will show the training set data and the best fitting line

-> call method scatter by plt library and give independent and dependent training set
-> call method plot by plt library and give training set of independent and prediction set
-> call title  method by plt library and give the title of graph
-> call xlabel method and tlabel method to give x axis and y axix label
-> call show method to show the graph   

"""

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

# visualising the test set results

"""

note : it will show the training set data and the best fitting line

-> call method scatter by plt library and give independent and dependent test set
-> call method plot by plt library and give training set of independent and prediction set
-> call title  method by plt library and give the title of graph
-> call xlabel method and tlabel method to give x axis and y axix label
-> call show method to show the graph   

"""

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

"""
note : finally analyze the graph 
-> accurate prediction (close line)
-> inaccurate prediction ()

note : the distance between y(test) and y(pred) (line) parallel to y axis
-> sum(y(test) - y(pred))^2  ==> minimum

"""

