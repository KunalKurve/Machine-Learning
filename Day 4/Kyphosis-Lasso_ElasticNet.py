# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:25:31 2024

@author: Administrator

"""

import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

kyp = pd.read_csv('Kyphosis.csv')
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

lr = LogisticRegression(solver = 'saga')
lasso = Lasso()
ridge = Ridge()
elastic = ElasticNet()

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

params = { 'penalty': ['elasticnet','L1','L2',None],
           'C':np.linspace(0.001, 10, 5), 
           'l1_ratio':np.linspace(0.001, 1, 4)}

gcv = GridSearchCV (lr, param_grid = params, cv = kfold, scoring = 'neg_log_loss')
gcv.fit(X, y)

pd_cv = pd.DataFrame( gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)

################################ Lasso ########################################

params = { 'alpha':np.linspace(0.001, 100, 50)}

gcv = GridSearchCV (lasso, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X, y)

############################## ElasticNet #####################################

params = { 'alpha':np.linspace(0, 100, 10),
           'l1_ratio':np.linspace(0, 1, 10)}

gcv = GridSearchCV (elastic, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X, y)

pd_cv = pd.DataFrame (gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)
