
# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DecisionTreeRegressor:
    def __init__(self, question, true_branch, false_branch, rows, column, value):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.row = row
        self.column = column
        self.value = value

    def is_numeric(value):
        return isinstance(value, int) or isinstance(value, float)

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def unique_vals(rows, col):
        return set([row[col] for row in rows])
    
    def class_counts(rows):
        counts = {}  
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
        
    def partition(rows, question):
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
    
    
    
    def gini(rows):
        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity
    
    def info_gain(left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * gini(left) - (1 - p) * gini(right)
    
    
    
    def find_best_split(rows):
        best_gain = 0  
        best_question = None  
        current_uncertainty = gini(rows)
        n_features = len(rows[0]) - 1  
    
        for col in range(n_features):  
            values = set([row[col] for row in rows])  
            for val in values: 
                question = Question(col, val)
                true_rows, false_rows = partition(rows, question)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gain = info_gain(true_rows, false_rows, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question = gain, question
        return best_gain, best_question
    

    def predict():
        if isinstance(node, Leaf):
            return node.predictions
        if node.question.match(row):
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)

    def fit(X, y):
        gain, question = find_best_split(X, y)
        if gain == 0:
            return Leaf(X, y)
        true_rows, false_rows = partition(X, y, question)
        true_branch = build_tree(true_rows)
        false_branch = build_tree(false_rows)
        return Decision_Node(question, true_branch, false_branch)
        

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

"""
note : it will divide the graph in region with respect to training set observation

-> import the decision tree regressor by sklearn library
-> make the regressor object with parameter random state = 0 by calling constructor
-> fit the regressor model in independent and dependent variable 

"""

"""
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
"""

regressor = DecisionTreeRegressor()
regressor.fit(X, y)


# Predicting a new result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

y_pred = regressor.predict(X_grid[55:56, ])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()