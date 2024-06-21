"""
Created on Mon May  6 16:53:21 2024

Cirrosis Stack Ensembling using LinR, XGB, CatB, LightGBM, ELasticNet.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from warnings import filterwarnings
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import StackingClassifier
import os

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Day 9\Multi-Class Prediction of Cirrhosis Outcomes")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv", index_col=0)
print(test.isnull().sum().sum())

le = LabelEncoder()

X_train = train.drop(['Status'],axis=1)
y_train = le.fit_transform(train['Status'])

X_dum = pd.get_dummies(X_train, drop_first=True)
test_dum = pd.get_dummies(test, drop_first=True)

en = ElasticNet()
cat = CatBoostClassifier(random_state=24)
lr = LogisticRegression()
xgb = XGBClassifier(random_state = 24)
lgb = LGBMClassifier(random_state = 24)

####################################### Stack Ensembling ########################################

stack = StackingClassifier([('LGB',lgb), ('LR',lr), ('CAT',cat), ('EN', en), ('XB',xgb)], 
                           final_estimator = lgb, passthrough= True)

# kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

stack.fit(X_dum, y_train)
# 
y_pred_prob1 = stack.predict_proba(test_dum)

submit1 = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob1[:,0],
                       'Status_CL':y_pred_prob1[:,1],
                       'Status_D':y_pred_prob1[:,2]})

print("For StackEnsembling")
print(submit1)

submit1.to_csv('StackEnsembling-Flood.csv',index=False)

####################################### Stack Ensembling using GSCV ########################################

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

params = {'final_estimator__learning_rate': np.linspace(0.001, 0.9, 5),
          'XB__max_depth': [None, 2]
          }

gscv = GridSearchCV(stack, param_grid=params, cv=kfold, scoring="neg_log_loss")

gscv.fit(X_dum, y_train)

dum_tst2 = pd.get_dummies(test, drop_first=True)

y_pred_prob2 = gscv.predict_proba(dum_tst2)

submit2 = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob2[:,0],
                       'Status_CL':y_pred_prob2[:,1],
                       'Status_D':y_pred_prob2[:,2]})

print("For StackEnsembling using GSCV")
print(submit2)

submit2.to_csv('StackEnsembling-Flood-GSCV.csv',index=False)

