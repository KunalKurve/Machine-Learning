# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:48:07 2024

@author: Administrator
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

kyp = pd.read_csv('Kyphosis.csv')
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
knn = KNeighborsClassifier(n_neighbors=5)

############################### Without Scaling ###############################

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=24, stratify = y)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)
print("Without Scaling")
print("Accuracy Score without Scaling: ", accuracy_score(y_test, y_pred))
print("ROC AUC Score without Scaling: ", roc_auc_score(y_test, y_pred_prob[:,1]))
print("Log Loss without Scaling: ", log_loss(y_test, y_pred_prob))

############################### With STD Scaling ##############################

std_scl = StandardScaler()

X_scl_train = std_scl.fit_transform(X_train)
X_scl_test = std_scl.fit_transform(X_test)

knn.fit(X_scl_train, y_train)

y_pred = knn.predict(X_scl_test)
y_pred_prob = knn.predict_proba(X_scl_test)

print("With Standard Scaling")
print("Accuracy Score with Scaling: ", accuracy_score(y_test, y_pred))
print("ROC AUC Score with Scaling: ", roc_auc_score(y_test, y_pred_prob[:,1]))
print("Log Loss with Scaling: ", log_loss(y_test, y_pred_prob))

############################### With MinMax Scaling ###############################

scl_m = MinMaxScaler()

X_minmax_train = scl_m.fit_transform(X_train)
X_minmax_test = scl_m.fit_transform(X_test)

knn.fit(X_minmax_train, y_train)

y_pred = knn.predict(X_minmax_test)
y_pred_prob = knn.predict_proba(X_minmax_test)

print("With MinMax Scaling")
print("Accuracy Score with MinMax Scaling: ", accuracy_score(y_test, y_pred))
print("ROC AUC Score with MinMax Scaling: ", roc_auc_score(y_test, y_pred_prob[:,1]))
print("Log Loss with MinMax Scaling: ", log_loss(y_test, y_pred_prob))

############################# Using Pipeline ##################################

knn = KNeighborsClassifier()

pipe_std = Pipeline([('SCL', std_scl) , ('KNN', knn)])

params = {'KNN__n_neighbors': [1,2,3,4,5,6,7,8,9,10]}

gcv = GridSearchCV (pipe_std, param_grid = params, cv = kfold, scoring = 'neg_log_loss')
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)
