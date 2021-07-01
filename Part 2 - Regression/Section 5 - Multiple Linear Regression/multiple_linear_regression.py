# Multiple Linear Regression

"""

note : multiple linear regression model

-> it possess one dependent and (>1) independent variable
-> y = a + b*x + c*x' + d*x'' + e*x''' ...
-> y = dependent variable && x = independent variable
-> a = slope && b = constant

"""

# Importing the libraries
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


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap

"""

note : In multiple linear regression model 
-> it possess categorical variable called dummy variable
-> never include all the dummy variables in equation
-> ever include number of dummy variable = n-1
-> because the coffecient of the that one dummy variable included in constant a

note : dummy variable (n) = 1 - dummy variable (n-1)

note : generally python library take care of the dummy variable and delete 0 coloumn

"""

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


mean_value = sum(y_train)/len(y_train)
max_value = max(y_train)

y_train = (y_train - mean_value)/max_value

"""
note : data preprocessing method
1) train and test split
2) feature scaling
3) encoding
    
"""
    

# Fitting Multiple Linear Regression to the Training set

"""
note : it will draw a line that best fits the graph (dataset point)

-> import linear regression class by linear_model of sklearn library
-> make object regressor of the class and call linear regression method
-> fit the regressor object in the independent and dependent training set 

note : it will take care of all the variable and it will be in (>1) dimension
-> it !can plot graph as it is 5 dimensional

"""

# from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results

"""

note : it will predict the value of dependent variable for test set values of independent variable 

-> call predict method via regressor object (give independent test set data)
-> it will return the vector of independent varaiable 
-> it will give (dependent value of best fitting line on independent variable) 

"""

y_pred = regressor.predict(X_test)

y_pred = y_pred*max_value + mean_value

"""
note : 
-> some independent variable statistically signifcant in prediction and some do !
-> if remove the non significant then it will give more accurate prediction

note : apply statsmodel library to evaluate the significance of independent variable
-> it do not consider a constant unlike other libraries

note : 
-> import statsmodel library
-> call ones method of np library (it will include the ones (array axis = 1) (50,1) in matrix of feature )     
1) it will add a ones coloumn
-> apply x optimal (first) it will possess all the coloumn of matrix of features
-> then it will filter one by one 
-> make object that will store the varaible returned by method OLS of sm library and fit it
1) endog = dependent variable and exog = optimal independent variable
-> call summary method by object then it will give a very important values
-> check the variable that possess p value higher that level
-> apply x optiaml again and remove that variable coloumn
    
"""

import statsmodels.api as sm

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_opt, X_test_opt = train_test_split(X_opt, test_size = 0.2, random_state = 0)


regressor.fit(X_train_opt,y_train)
y_opt = regressor.predict(X_test_opt)

