# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:30:11 2024

@author: Administrator
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

kyp = pd.read_csv('Kyphosis.csv')
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)


############################################### KNN ########################################

# Without Scaling

knn = KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=24, stratify = y)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred_prob[:,1]))
print("Log Loss: ", log_loss(y_test, y_pred_prob))
print()

############################################# GSCV ############################################

knn = KNeighborsClassifier()
params = {'n_neighbors':[1,2,3,4,5,6,7,8]}
gcv = GridSearchCV (knn, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)
