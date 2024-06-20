# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:41:08 2024

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

glass = pd.read_csv("Glass.csv")
le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis=1)

lr = LogisticRegression (solver = 'saga', random_state = 24)

kfold = StratifiedKFold (n_splits = 5, shuffle = True, random_state = 24)

params = { 'penalty': ['elasticnet', 'L1', 'L2', None],
           'C':np.linspace(0.001, 10, 5), 
           'l1_ratio':np.linspace(0.001, 1, 4),
           'multi_class': ['ovr', 'multinomial']}

gcv = GridSearchCV (lr, param_grid = params, cv = kfold, scoring = 'neg_log_loss')
gcv.fit(X, y)

pd_cv = pd.DataFrame( gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)
