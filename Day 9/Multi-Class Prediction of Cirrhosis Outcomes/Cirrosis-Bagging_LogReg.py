import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from warnings import filterwarnings
import os

filterwarnings('ignore')

os.chdir("D:\March 2024\PML\Day 7\Multi-Class Prediction of Cirrhosis Outcomes")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv", index_col=0)
print(test.isnull().sum().sum())

le = LabelEncoder()

X_train = pd.get_dummies(train.drop('Status', axis=1), drop_first=True)
y_train = le.fit_transform(train['Status'])

X_test = test.drop('id',axis=1)

lr = LogisticRegression()
lr.fit(X_train,y_train)

dum_tst = pd.get_dummies(test, drop_first=True)

y_pred_prob = lr.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})

print(submit)

# submit.to_csv("Cirrosis_bagging-LogReg.csv", index=False)

params = {'estimator__min_samples_split':np.arange(2,35,5),
          'estimator__min_samples_leaf':np.arange(1, 35, 5),
          'estimator__max_depth':[None, 4, 3, 2]}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state = 24)

dtc = DecisionTreeClassifier(random_state = 24)

bagg = BaggingClassifier(dtc, random_state = 24)

print()

gcv = GridSearchCV(bagg, param_grid=params, verbose=3, cv=kfold, scoring="neg_log_loss", n_jobs=-1)

gcv.fit(X_train, y_train)

print(gcv.best_params_)
print(gcv.best_score_)

dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = gcv.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("Cirrosis1_bagging.csv", index=False)