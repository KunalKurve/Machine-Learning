"""
2. Consider the dataset at the link Santander Customer Satisfaction | Kaggle.
Do the following:
Find out how many principal components explain more than 90% variation taking 
all the variables except ID and target. 
Try the following models with PCA transform (Pipeline):
- Random Forest
- X G Boost
- Cat Boost
- Light GBM
Mention the leaderboard scores if possible
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
from warnings import filterwarnings


filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Assignment\Assignment 2 - Santander Customer Satisfaction")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())

test = pd.read_csv("test.csv", index_col=0)
print(test.isnull().sum().sum())

X_train = train.drop(['TARGET'], axis = 1)
y_train = train['TARGET']
X_test = test.copy()

scaler = StandardScaler().set_output(transform='pandas')
train_scl = scaler.fit_transform(train)

pca = PCA().set_output(transform='pandas')
principalComponents = pca.fit_transform(train_scl)

#####################################################################################################################

var = np.cumsum(pca.explained_variance_ratio_ * 100)
print("Variance:")
print(var)
count = (var > 90).sum()
print("\nNo. of features is:", var.sum())
print("\nNo. of features that give varaiation more than 90% is:", count)

########################################## RandomForestClassifier #####################################################

rfc = RandomForestClassifier(random_state=24)

pipe_rfc = Pipeline([('SCL', scaler), ('PCA', pca), ('TREE', rfc)]) 

pipe_rfc.fit(X_train, y_train)

y_pred_rfc = pipe_rfc.predict(X_test)
y_pred_prob_rfc = pipe_rfc.predict_proba(X_test)
y_pred_prob_rfc = y_pred_prob_rfc [:,1]

submit_rfr = pd.DataFrame({'ID': list(test.index), 'TARGET': y_pred_prob_rfc})

print("\nFor RandomForest Regressor")
print(submit_rfr)

submit_rfr.to_csv('Santander-RandomForestClassifier.csv', index=False)

################################################ XGBClassifier ########################################################

xgb = XGBClassifier()

pipe_xgb = Pipeline([('SCL', scaler), ('PCA', pca), ('XGB', xgb)]) 

pipe_xgb.fit(X_train, y_train)

y_pred_xgb = pipe_xgb.predict(X_test)
y_pred_prob_xgb = pipe_xgb.predict_proba(X_test)
y_pred_prob_xgb = y_pred_prob_xgb [:,1]

submit_xgb = pd.DataFrame({'ID': list(test.index), 'TARGET': y_pred_prob_xgb})

print("\nFor XGB Classifier")
print(submit_xgb)

submit_xgb.to_csv('Santander-XGBClassifier.csv', index=False)

################################################ CatBoostClassifier ########################################################

cat = CatBoostClassifier()

pipe_cat = Pipeline([('SCL', scaler), ('PCA', pca), ('CAT', cat)]) 

pipe_cat.fit(X_train, y_train)

y_pred_cat = pipe_cat.predict(X_test)
y_pred_prob_cat = pipe_cat.predict_proba(X_test)
y_pred_prob_cat = y_pred_prob_cat [:,1]

submit_cat = pd.DataFrame({'ID': list(test.index), 'TARGET': y_pred_prob_cat})

print("\nFor CatBoost Classifier")
print(submit_cat)

submit_cat.to_csv('Santander-CatBoostClassifier.csv', index=False)

################################################ LGBMClassifier ########################################################

lgmb = LGBMClassifier()

pipe_lgmb = Pipeline([('SCL', scaler), ('PCA', pca), ('LGBM', lgmb)]) 

pipe_lgmb.fit(X_train, y_train)

y_pred_lgmb = pipe_lgmb.predict(X_test)
y_pred_prob_lgmb = pipe_lgmb.predict_proba(X_test)
y_pred_prob_lgmb = y_pred_prob_lgmb [:,1]

submit_lgbm = pd.DataFrame({'ID': list(test.index), 'TARGET': y_pred_prob_lgmb})

print("\nFor LGBMClassifier")
print(submit_lgbm)

submit_lgbm.to_csv('Santander-LGBMClassifier.csv', index=False)

################################################ All ########################################################

pipe = [pipe_rfc, pipe_xgb, pipe_cat, pipe_lgmb] 

pipe_lgmb.fit(X_train, y_train)

y_pred_lgmb = pipe_lgmb.predict(X_test)
y_pred_prob_lgmb = pipe_lgmb.predict_proba(X_test)
y_pred_prob_lgmb = y_pred_prob_lgmb [:,1]

submit_lgbm = pd.DataFrame({'ID': list(test.index),
                           'TARGET': y_pred_prob_lgmb})

print("\nFor LGBMClassifier")
print(submit_lgbm)
  
kfold = KFold(n_splits = 5, shuffle = True, random_state=24)
imp = SimpleImputer()
lr = LinearRegression()

pipe = Pipeline([('IMP',imp), ("LR", lr)])

params = {'IMP__strategy':['mean', 'median', 'most-frequent']}

gcv = GridSearchCV(pipe, param_grid = params, cv = kfold)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

# submit_lgbm.to_csv('Santander-LGBMClassifier.csv', index=False)