# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:49:46 2024

@author: Administrator
"""
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

meds = pd.read_csv('insurance.csv')

dum = pd.get_dummies(meds, drop_first=True)

y = dum['charges']
X = dum.drop('charges', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24)

################################ Ridge ########################################

ridge = Ridge(alpha=0.02)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print("\nR2 Score for Ridge")
print(r2_score(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24)

################################ Lasso ########################################

lasso = Lasso(alpha=0.02)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
print("\nR2 Score for Lasso")
print(r2_score(y_test, y_pred))

# K-FOLD #

kfold = KFold(n_splits=5, shuffle=True, 
              random_state=24)
lambdas = np.linspace(0.001, 100,40)
scores = []
for i in lambdas:
    ridge = Lasso(alpha=i)
    results = cross_val_score(ridge, X, y,
                              cv=kfold)
    scores.append(results.mean())

i_max = np.argmax(scores)
print("\nKFold")
print("Best alpha =", lambdas[i_max])

############################# GridSearch ######################################

params = {'alpha':np.linspace(0.001, 100,40)}
gcv = GridSearchCV(ridge, param_grid=params,
                   cv=kfold)
gcv.fit(X, y)
print("\nR2 Score for GridSearch")
print(gcv.best_params_)
print(gcv.best_score_)

############################# ElasticNet ######################################

elastic = ElasticNet()
print(elastic.get_params())
params = {'alpha':np.linspace(0.001, 50, 5), 
          'l1_ratio':np.linspace(0.001, 1, 10)}

gcv = GridSearchCV(elastic, param_grid=params,
                   cv=kfold)
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_)
print("\nR2 Score for ElasticNet")
print(gcv.best_params_)
print(gcv.best_score_)