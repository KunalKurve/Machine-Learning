# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:23:26 2024

@author: Administrator
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

boston = pd.read_csv('Boston.csv')
print(boston)

X = boston.drop('medv', axis=1)
y = boston['medv']

# Why do we put '[[]]'?
# X should be 2D Numpy Array to perform dot fit. y can be 1D or 2D array

poly1 = PolynomialFeatures (degree = 1).set_output(transform = "pandas")
X_poly1 = poly1.fit_transform(X)

lr = LinearRegression()

lr.fit(X_poly1, y)

print(lr.coef_)
print(lr.intercept_)



poly2 = PolynomialFeatures (degree = 2).set_output(transform = None)
X_poly1 = poly1.fit_transform(X)

lr = LinearRegression()

lr.fit(X_poly1, y)

print(lr.coef_)
print(lr.intercept_)




poly1 = PolynomialFeatures (degree = 1).set_output(transform = "pandas")
X_poly1 = poly1.fit_transform(X)

lr = LinearRegression()

lr.fit(X_poly1, y)

print(lr.coef_)
print(lr.intercept_)

