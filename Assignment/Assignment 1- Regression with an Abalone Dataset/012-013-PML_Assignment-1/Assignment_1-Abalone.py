'''
Consider the Kaggle Competition at the link https://www.kaggle.com/competitions/playground-series-s4e4 on Abalone.
Try the following models:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Stacking with models of your choice

Mention leaderboard scores for each of the five.
'''

import pandas as pd
import numpy as np
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
import os
from warnings import filterwarnings

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Assignment\Assignment 1- Regression with an Abalone Dataset")

train = pd.read_csv("train.csv", index_col=0)
train = train.drop(['Sex'],axis=1)
print(train.isnull().sum().sum())

test = pd.read_csv("test.csv", index_col=0)
test = test.drop(['Sex'],axis=1)
print(test.isnull().sum().sum())

X_train = train.drop(['Rings'], axis = 1)
y_train = train['Rings']

X_test = test.copy()

lr = LinearRegression()
rfr = RandomForestRegressor()
xgb = XGBRFRegressor()
lgbm = LGBMRegressor()
cat = CatBoostRegressor()

################################## Linear Regression ###########################

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_lr_ceil = np.ceil(y_pred_lr)
y_pred_lr_ceil[y_pred_lr_ceil < 0] = 0

submit_lr = pd.DataFrame({'id': list(test.index),
                          'Rings': y_pred_lr_ceil})

print("For Linear Regression")
print(submit_lr)

submit_lr.to_csv("Abalone-LinearRegression.csv", index=False)

################################ RandomForestRegressor #########################

rfr.fit(X_train, y_train)

y_pred_rfr = rfr.predict(X_test)
y_pred_rfr_ceil = np.ceil(y_pred_rfr)
y_pred_rfr_ceil[y_pred_rfr_ceil < 0] = 0

submit_rfr = pd.DataFrame({'id': list(test.index),
                          'Rings': y_pred_rfr_ceil})

print("\nFor RandomForest Regressor")
print(submit_rfr)


submit_rfr.to_csv("Abalone-RandomForestRegression.csv", index=False)

################################## XGBRF Regressor ###########################

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_pred_xgb_ceil = np.ceil(y_pred_xgb)
y_pred_xgb_ceil[y_pred_xgb_ceil < 0] = 0

submit_xgb = pd.DataFrame({'id': list(test.index),
                          'Rings': y_pred_xgb_ceil})

print("\nFor XGBRF Regressor")
print(submit_xgb)

submit_xgb.to_csv("Abalone-XGBRFRegression.csv", index=False)

################################### LGBMRegressor #############################

lgbm.fit(X_train, y_train)

y_pred_lgbm = lgbm.predict(X_test)
y_pred_lgbm_ceil = np.ceil(y_pred_lgbm)
y_pred_lgbm_ceil[y_pred_lgbm_ceil < 0] = 0

submit_lgbm = pd.DataFrame({'id': list(test.index),
                          'Rings': y_pred_lgbm_ceil})

print("\nFor LGBM Regressor")
print(submit_lgbm)

submit_lgbm.to_csv("Abalone-LGBMRegression.csv", index=False)

############################## Stack Ensembling ###############################

stack = StackingRegressor([('LR',lr), ('TREE',rfr), ('XGB',xgb), ('LGBM', lgbm)], 
                           final_estimator = cat, passthrough= True)

stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)
y_pred_ceil = np.ceil(y_pred_stack)
y_pred_ceil[y_pred_ceil < 0] = 0
submit_stack = pd.DataFrame({'id': list(test.index),
                          'Rings': y_pred_ceil})

print("For Stack Ensembling")
print(submit_stack)

submit_stack.to_csv("Abalone-StackEnsembling.csv", index=False)
