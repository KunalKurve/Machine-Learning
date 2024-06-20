# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:12:13 2024

@author: Administrator
"""

import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import log_loss
# from sklearn.metrics import r2_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


kyp = pd.read_csv("Kyphosis.csv")
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis = 1)

lr = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24, stratify=y)


lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)

print("Accuracy Score: ",accuracy_score(y_test, y_pred))
print("Log Loss: ", log_loss(y_test, y_pred_prob))
print("ROC and AUC Score: ",roc_auc_score(y_test, y_pred_prob[:,1]))

#Doing this same with Bankruptcy dataset gives
# Convergence warning

bank =  pd.read_csv("Bankruptcy.csv")


X = bank.drop(['D','NO'], axis=1)
y = bank['D']


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)


lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)

############## Model Evaluation ##############
print("Accuracy Score: ",accuracy_score(y_test, y_pred))
print("Log Loss: ", log_loss(y_test, y_pred_prob))
print("ROC and AUC Score: ",roc_auc_score(y_test, y_pred_prob[:,1]))