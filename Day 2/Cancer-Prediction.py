# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:10:57 2024

@author: Administrator
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import log_loss
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


cancer =  pd.read_csv("Cancer.csv")


# le = LabelEncoder()
# y = le.fit_transform(cancer['Class'])
# X = cancer.drop('Class', axis=1)
# print(le.classes_)

# #or

dum_can = pd.get_dummies(cancer, drop_first=True)
y = dum_can['Class_recurrence-events']
X = dum_can.drop(['Class_recurrence-events', 'subjid'], axis = 1)

bnb = BernoulliNB()

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=2021,
                                                    stratify=y)

bnb.fit(X_train, y_train)

y_pred = bnb.predict(X_test)
y_pred_prob = bnb.predict_proba(X_test)

############## Model Evaluation ##############
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy Score: ",accuracy_score(y_test, y_pred))
print("Log Loss: ", log_loss(y_test, y_pred_prob))
print("ROC and AUC Score: ",roc_auc_score(y_test, y_pred_prob[:,1]))
# Compute predicted probabilities: y_pred_prob
y_probs = bnb.predict_proba(X_test)

print("R squared ", r2_score(y_test, y_pred))
############## Model Evaluation ##############
Skfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

############## Model Evaluation ##############

result = cross_val_score(bnb, X, y, cv = Skfold)
print(result.mean())

results = cross_val_score(bnb, X, y, cv = Skfold, scoring='roc_auc')
print(results.mean())

results = cross_val_score(bnb, X, y, cv = Skfold, scoring = 'neg_log_loss')
print(results.mean())



