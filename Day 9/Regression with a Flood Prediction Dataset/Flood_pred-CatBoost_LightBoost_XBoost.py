import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
from warnings import filterwarnings
import os

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Day 9\Regression with a Flood Prediction Dataset")
train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis = 1)
y_train = train['FloodProbability']

X_test = test.drop('id', axis = 1)

####################################### CatBoostRegressor ########################################

cbr = CatBoostRegressor(random_state=24)

# kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

# params = {'learning_rate': np.linspace(0.001, 0.9, 5),
#           'max_depth': [None, 2],
#           'n_estimators': [25, 50]}

# gsv = GridSearchCV (cbr, param_grid=params, cv=kfold, scoring='r2')

cbr.fit(X_train, y_train)
y_pred = cbr.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'Flood Probability':y_pred})
print(submit)

submit.to_csv('CatBoostRegressor-Flood.csv',index=False)

####################################### XBoostRegressor ########################################

xgb = XGBRFRegressor(random_state=24)

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

params = {'learning_rate': np.linspace(0.001, 0.9, 5),
          'max_depth': [None, 2],
          'n_estimators': [25, 50]}

gsv = GridSearchCV (xgb, param_grid=params, cv=kfold, scoring='r2')

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'Flood Probability':y_pred})

print(submit)
# print(gsv.best_params_)
# print(gsv.best_score_)

# submit.to_csv('XGBRFRegressor-Flood.csv',index=False)
submit.to_csv('XGBRFRegressor-Flood-GCV.csv',index=False)

####################################### LightGradientBoostRegressor ########################################

lgb = LGBMRegressor(random_state=24)

# kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

# params = {'learning_rate': np.linspace(0.001, 0.9, 5),
#           'max_depth': [None, 2],
#           'n_estimators': [25, 50]}

# gsv = GridSearchCV (lgb, param_grid=params, cv=kfold, scoring='r2')

lgb.fit(X_train, y_train)
y_pred = lgb.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'Flood Probability':y_pred})

print(submit)

submit.to_csv('LGBMRegressor-Flood.csv',index=False)
