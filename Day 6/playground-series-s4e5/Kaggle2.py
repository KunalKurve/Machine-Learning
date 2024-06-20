import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import os

os.chdir(r"D:\March 2024\PML\playground-series-s4e5")
train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis = 1)
y_train = train['FloodProbability']

X_test = test.drop('id', axis = 1)
# y_test = test['id']

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

el = ElasticNet()

# print(pipe.get_params())

params = {'alpha': np.linspace(0.001, 5, 3),
          'l1_ratio': np.linspace(0.001, 1, 3)}
gcv_el = GridSearchCV(el, param_grid = params, cv = kfold, scoring='r2')

gcv_el.fit(X_train, y_train)
y_pred = gcv_el.predict(X_test)

submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability': y_pred})
print(submit)
submit.to_csv("submit-EL.csv", index=False)