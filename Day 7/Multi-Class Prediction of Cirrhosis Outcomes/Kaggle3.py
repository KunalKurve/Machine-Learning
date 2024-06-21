'''
This approch is used for Multilabel Classification, and not for Multiclass Classification. 
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

os.chdir(r"D:\March 2024\PML\Day 7\Multi-Class Prediction of Cirrhosis Outcomes")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())

train_dum = pd.get_dummies(train)
test_dum = pd.get_dummies(test)

X_test = test_dum.drop('id', axis = 1)

le = LabelEncoder()

params = {'min_samples_split': np.arange(2,35,5),
          'min_samples_leaf': np.arange(1,35,5),
          'max_depth': [None, 4,3,2]}

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

dtc = DecisionTreeClassifier(random_state=24)

gcv = GridSearchCV(dtc, param_grid = params, cv = kfold, scoring='neg_log_loss')

#################################### CL ###########################################

X_train_CL = train_dum.drop(['Status_CL','Status_D','Status_C'], axis = 1)
y_train_CL = le.fit_transform(train_dum['Status_CL'])

gcv.fit(X_train_CL, y_train_CL)
y_pred_CL = gcv.predict(X_test)
y_pred_prob_CL = pd.DataFrame(gcv.predict_proba(X_test))
y_pred_prob_CL_max = y_pred_prob_CL[[0,1]].min(axis = 1)
print("For Status CL")
print(gcv.best_params_)
print(gcv.best_score_)

#################################### D ###########################################

X_train_D = train_dum.drop(['Status_CL','Status_D','Status_C'], axis = 1)
y_train_D = le.fit_transform(train_dum['Status_D'])

gcv.fit(X_train_D, y_train_D)
y_pred_D = gcv.predict(X_test)
y_pred_prob_D = pd.DataFrame(gcv.predict_proba(X_test))
y_pred_prob_D_max = y_pred_prob_D[[0,1]].min(axis = 1)
print("For Status D")
print(gcv.best_params_)
print(gcv.best_score_)

#################################### C ###########################################

X_train_C = train_dum.drop(['Status_CL','Status_D','Status_C'], axis = 1)
y_train_C = le.fit_transform(train_dum['Status_C'])

gcv.fit(X_train_C, y_train_C)
y_pred_C = gcv.predict(X_test)
y_pred_prob_C = pd.DataFrame(gcv.predict_proba(X_test))
y_pred_prob_C_max = y_pred_prob_C[[0,1]].min(axis = 1)
print("For Status C")
print(gcv.best_params_)
print(gcv.best_score_)

##################################################################################

best_tree = gcv.best_estimator_

submit = pd.DataFrame({'id':test['id'],
                       'Status_C': y_pred_prob_C_max,
                       'Status_CL': y_pred_prob_CL_max,
                       'Status_D': y_pred_prob_D_max})
print(submit)
submit.to_csv("CirrosisSubmit.csv", index=False)
