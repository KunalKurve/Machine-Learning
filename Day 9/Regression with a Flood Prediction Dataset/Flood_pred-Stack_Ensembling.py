import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from warnings import filterwarnings
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import StackingRegressor
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

en = ElasticNet()
cat = CatBoostRegressor()
lr = LinearRegression()
xgb = XGBRFRegressor (random_state = 24)

lgb = LGBMRegressor (random_state = 24)

stack = StackingRegressor([('LGB',lgb), ('LR',lr), ('CAT',cat), ('EN', en), ('XB',xgb)], final_estimator = lgb)

####################################### Stack Ensembling ####################################

stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'Flood Probability':y_pred})
print(submit)

submit.to_csv("Flood_Prediction-Stack_Ensembling(lgb-Lin-cat-EN-XGB;Final-LGB).csv", index = False)

################################# GSCV #########################################

# kfold = KFold(n_splits=5, shuffle= True, random_state= 24)

# params = {'EN__alpha': np.linspace(0.001, 3, 5),
#           'XB__max_depth': [None, 2],
#           'XB__gamma': np.linspace(0.001, 3, 5),
#           'passthrough': [True, False]}

# gscv = GridSearchCV(stack, param_grid = params, cv = kfold, scoring='r2')

# gscv.fit(X_train,y_train)
# y_pred = gscv.predict(X_test)

# submit = pd.DataFrame({'id':test['id'],'Flood Probability':y_pred})
# print(submit)