# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:42:43 2024

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

boston = pd.read_csv('Boston.csv')
print(boston)

X = boston.drop('medv', axis = 1)
y = boston['medv']

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)
lr = LinearRegression()

############################# Using Loop ######################################

degrees = [1,2,3,4,5]
scores = []

# poly = PolynomialFeatures (degree = i).set_output(transform = "pandas") - 
# did not write .set_output because it takes a lot of time to compute using pandas. 
# it is faster with numpy

for i in degrees:
    poly = PolynomialFeatures (degree = i)  
    pipe = Pipeline([("POLY",poly) , ('LR',lr)])
    results = cross_val_score(pipe, X, y, cv = kfold)
    scores.append(results.mean())
    
i_max = np.argmax(scores)
print("Best Degree (Loop):", degrees[i_max])
print("Best Score (Loop):", scores[i_max])

######################### Without GridSearch ##################################

degrees = [1,2,3,4,5]

print(pipe.get_params())
params = {'POLY__degree':[1,2,3,4,5]}
gcv = GridSearchCV(pipe, param_grid=params, cv = kfold)
gcv.fit(X, y)

print("Best Degree (GridSearch):", gcv.best_score_)
print("Best Score (GridSearch):", gcv.best_params_)

########################### Ridge Transform ###################################

rd = Ridge()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY', poly),('RIDGE', rd)])















