# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:09:30 2024

@author: Administrator
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pizza = pd.read_csv('pizza.csv')
print(pizza)

X = pizza[['Promote']]
y = pizza['Sales']

# Why do we put '[[]]'?
# X should be 2D Numpy Array to perform dot fit. y can be 1D or 2D array

poly = PolynomialFeatures (degree = 2).set_output(transform = None)
X_poly = poly.fit_transform(X)

lr = LinearRegression()

lr.fit(X_poly, y)

print(lr.coef_)
print(lr.intercept_)