# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:41:23 2024

@author: Administrator

Ridge Regression:
Z = ∑(yi - yi_bar)^2 + α ∑bj^2
α - in sklearn
λ - in Stats
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

conc =  pd.read_csv("Concrete_Data.csv")

X = conc.drop('Strength', axis=1)
y = conc['Strength']


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=2021)

rg = Ridge()
rg.fit(X_train, y_train)

y_pred = rg.predict(X_test)
print(r2_score(y_test, y_pred))

############## Model Evaluation ##############

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)
lambdas = np.linspace(0, 30, 20)
scores = []

for i in lambdas:
    rg = Ridge(alpha = i)
    results = cross_val_score(rg, X, y, cv = kfold)
    scores.append(results.mean())

i_min = np.argmin(scores)
print(i_min)
print("Worst Alpha: ", lambdas[i_min])

i_max = np.argmax(scores)
print(i_max)
print("Best Alpha: ", lambdas[i_max])
e_max = scores[i_max]
print(e_max)
print(np.max(scores))

