import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
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
std_slr = StandardScaler()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=24)

flood_pred = VotingRegressor([('LR',lr), ('Ridge', ridge), ('Lasso',lasso),('TREE',dtr)])

kfold = KFold(n_splits=5, random_state=24, shuffle=True)

params = {'Ridge__alpha': np.linspace(0.001, 3, 5),
          'Lasso__alpha': np.linspace(0.001, 3, 5),
          'TREE__max_depth': [None, 3, 4, 5],
          'TREE__min_samples_leaf': [2, 4, 5, 8],
          'TREE__min_samples_split': [1, 4, 5, 8]}

rcv = RandomizedSearchCV(flood_pred, param_distributions = params, random_state=24, 
                         cv = kfold,scoring = 'r2', n_jobs = -1, n_iter = 20)

rcv.fit(X_train,y_train)
print("\nUsing RSCV:")
print("R2 Score:",rcv.best_score_)
print("Best Parameters:\n",rcv.best_params_)

y_pred = rcv.predict(X_test)

submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability': y_pred})
print(submit)

submit.to_csv("Flood_Pred_1.csv", index=False)