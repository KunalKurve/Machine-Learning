# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:23:26 2024

@author: Administrator

Types of Transforms: 
    1. Polynomial Transformation


Note:- 
whenever we do train_test_split - for transformation function:
    .fit - to be applied on - (train set object)
    .transform - to be applied on - (train)
    .transform - to be applied on - (test)

Not to practice: 
    .fit(test set object)
    

"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

boston = pd.read_csv('Boston.csv')
print(boston)

X = boston.drop('medv', axis=1)
y = boston['medv']

# Why do we put '[[]]'?
# X should be 2D Numpy Array to perform dot fit. y can be 1D or 2D array

poly1 = PolynomialFeatures (degree = 1).set_output(transform = "pandas")
X_poly1 = poly1.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly1, y, test_size = 0.3, random_state=24)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

######################################################################################################

poly2 = PolynomialFeatures (degree = 2).set_output(transform = None)
X_poly2 = poly2.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly2, y, test_size = 0.3, random_state=24)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

######################################################################################################

poly3 = PolynomialFeatures (degree = 3).set_output(transform = "pandas")
X_poly3 = poly3.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly3, y, test_size = 0.3, random_state=24)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))