import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import BaggingRegressor
from warnings import filterwarnings
import os

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Day 8\Regression with a Flood Prediction Dataset")
train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis = 1)
y_train = train['FloodProbability']

X_test = test.drop('id', axis = 1)

lr = LinearRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=24)

params = {'estimator__min_samples_split':np.arange(2,35,5),
          'estimator__min_samples_leaf':np.arange(1, 35, 5),
          'estimator__max_depth':[None, 4, 3, 2]}

bagg = BaggingRegressor(lr, n_estimators=25, random_state=24) 

bagg.fit(X_train, y_train)
y_pred = bagg.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'Flood Probability':y_pred})

print(submit)

submit.to_csv('Bagging_Linear-regression.csv',index=False)

#################################################################################

el = ElasticNet()

bagg = BaggingRegressor(el, n_estimators=25, random_state=24)

params={'estimator__alpha': np.linspace(0.001,5,2),
       'estimator__l1_ratio':np.linspace(0,1,2)}

gcv = GridSearchCV(bagg, param_grid=params, cv=kfold)

gcv.fit(X_train, y_train)

best_model = gcv.best_esti

print(best_model)
