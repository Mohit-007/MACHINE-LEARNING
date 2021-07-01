# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:47:05 2019

@author: RAMESHWAR LAL JI
"""

"""note : save the data (.csv) file and python (.py) file together"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt # (it plot the graph)
import pandas as pd             # (it import the data)
 

"""
note : 
    1) train test split
    2) feature scaling
    3) encoding
    
"""



# importing the dataset

"""
note : 
-> (it possess the method read_csv that import the dataset)
-> (the variable object store the data of the file)
"""

"""
note : the differentiation of independent and dependent variable
-> independent variable (generally matrix)  (matrix of feature) 
-> dependent variable (generally vector)    
"""

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, 3].values

# (variable object) = dataset.iloc[a:b, x:y].values
 
"""
-> it will return row [a,b) && coloumn [x,y) values of the dataset in the object
-> the variable object (independent | dependent) store the matrix and vector returned  
-> if a & b (void) then it will possess all the row (general case)
-> write varaible object in console and get independent variable matrix and dependent vector
"""

# filling the missing data

"""
note :
-> if remove all row data of 1 missing data in a particular row or coloumn 
-> then it may possess crucial information that can be danger

-> import imputer class by preprocessing library in sklearn library
-> make object of the class and call imputer method and fill parameter
1) missing value = 'place value of value' (it will know the position of missing value)
2) strategy  = ('mean'|'median'|'most_frequent') (taking strategy to fill the value)
3) axis = 0|1 (applying the strategy by coloumn = 0 && row = 1)
-> it will fit the imputer object in a missing value coloumn or row of varaible object
-> it will transform the row and coloumn of variable object by imputer object 

note : In fit and transform pass the independent variable matrix 

"""

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# encoding the categorical data

"""
note :
-> the machine learning model work in the basis of math so it is difficult to manuplate string
-> the encoding transforms the string into the integer 


-> import the label encoder and one hot encoder class by preprocessing library of sklearn 

-> make object of the class and call label encoder method 
-> it will fit the imputer object in a missing value coloumn or row of varaible object
-> it will transform the missing row and coloumn of variable object by imputer object 

note : it is applicable if the model misunderstand the mathematical value of a coloumn

-> make object of the class and call one hot encoder method and fill parameter
1) categorical_features = [ coloumn ] 
-> it will fit the imputer object in all coloumn or row of varaible object
-> it will transform all the row and coloumn of variable object by imputer object 

"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0] )
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# splitting the dataset

"""
note : let's split the data in in training and testing 

-> import the train_test_split method by model_selection library of sklearn library
-> make variable => independent variable train and test && dependent variable train && test 
-> call the method and fill the parameter
1) give independent and dependent variable 
2) apply test size value (generally 0.25)
3) apply random_state = 0 (it generate identical conclusion)



"""

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# feature scaling

"""
note :
-> it is applied to get value of all the independent variable and dependent variable in scale

-> import standard_scaler class by preprocessing library of sklearn library
-> make object of standard_scaler class and call the method (independent)
-> fit and transform in training set object 
-> transform in test set object

note : it can be applied in the dependent variable (if required)  

"""

from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

