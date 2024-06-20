# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 08:26:08 2024

@author: Administrator
""" 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


sonar = pd.read_csv("Sonar.csv")

le = LabelEncoder()

y = le.fit_transform(sonar['Class'])
X = sonar.drop('Class', axis=1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state = 2021)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(r2_score(y_test, y_pred))

#G NB and Logistic Regression give error for the above case

############## Model Evaluation ##############
kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

############## Model Evaluation ##############

results = cross_val_score(gnb, X, y, cv = kfold)
print(results)
print(results.mean())

results = cross_val_score(gnb, X, y, cv = kfold, scoring='roc_auc')
print(results)
print(results.mean())