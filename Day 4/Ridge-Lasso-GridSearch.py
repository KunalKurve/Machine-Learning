# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:47:37 2024

@author: Administrator
"""
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

conc = pd.read_csv("Concrete_Data.csv")
X = conc.drop('Strength', axis=1)
y = conc['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24)

################################ Ridge ########################################

ridge = Lasso(alpha=0.02)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print(r2_score(y_test, y_pred))

kfold = KFold(n_splits=5, shuffle=True, 
              random_state=24)

lambdas = np.linspace(0.001, 100,40)
scores = []
################################ Lasso ########################################
for i in lambdas:
    ridge = Lasso(alpha=i)
    results = cross_val_score(ridge, X, y, cv=kfold)
    scores.append(results.mean())

i_max = np.argmax(scores)
print("Best alpha =", lambdas[i_max])

############################# GridSearch ######################################

params = {'alpha':np.linspace(0.001, 100,40)}
gcv = GridSearchCV(ridge, param_grid=params, cv=kfold)
gcv.fit(X, y)
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
print(gcv.best_params_)
print(gcv.best_score_)