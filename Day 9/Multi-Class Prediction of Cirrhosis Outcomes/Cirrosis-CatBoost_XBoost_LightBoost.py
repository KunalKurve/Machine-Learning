import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from warnings import filterwarnings
import os

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Day 9\Multi-Class Prediction of Cirrhosis Outcomes")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv", index_col=0)
print(test.isnull().sum().sum())

train_dum = pd.get_dummies(train)
test_dum = pd.get_dummies(test, drop_first=True)

le = LabelEncoder()

X = train.drop(['Status'],axis=1)
y = le.fit_transform(train['Status'])

X_dum = pd.get_dummies(X, drop_first=True)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

params = {'learning_rate': np.linspace(0.001, 0.9, 5),
          'max_depth': [None, 2],
          'n_estimators': [25, 50]}

####################################### CatBoostClassifier ########################################

cbc = CatBoostClassifier(random_state=24)

gcv = GridSearchCV(cbc, param_grid=params, cv=kfold, scoring="neg_log_loss")

gcv.fit(X_dum, y)

dum_tst1 = pd.get_dummies(test, drop_first=True)

y_pred_prob1 = gcv.predict_proba(dum_tst1)

submit1 = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob1[:,0],
                       'Status_CL':y_pred_prob1[:,1],
                       'Status_D':y_pred_prob1[:,2]})

print("For CatBoostClassifier")
print(submit1)

submit1.to_csv('CatBoostClassifier-Flood-GCV.csv',index=False)

####################################### XGBClassifier ########################################

xgb = XGBClassifier(random_state=24)

gcv = GridSearchCV(xgb, param_grid=params, cv=kfold, scoring="neg_log_loss")

gcv.fit(X_dum, y)

dum_tst2 = pd.get_dummies(test, drop_first=True)

y_pred_prob2 = gcv.predict_proba(dum_tst2)

submit2 = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob2[:,0],
                       'Status_CL':y_pred_prob2[:,1],
                       'Status_D':y_pred_prob2[:,2]})

print("For XGBClassifier")
print(submit2)

submit2.to_csv('XGBClassifier-Flood-GCV.csv',index=False)

####################################### LGBMClassifier ########################################

lgb = LGBMClassifier(random_state=24)

gcv = GridSearchCV(lgb, param_grid=params, cv=kfold, scoring="neg_log_loss")

gcv.fit(X_dum, y)

dum_tst3 = pd.get_dummies(test, drop_first=True)

y_pred_prob3 = gcv.predict_proba(dum_tst3)

submit3 = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob3[:,0],
                       'Status_CL':y_pred_prob3[:,1],
                       'Status_D':y_pred_prob3[:,2]})

print("For LGBMClassifier")
print(submit3)

submit3.to_csv('LGBMClassifier-Flood-GCV.csv',index=False)