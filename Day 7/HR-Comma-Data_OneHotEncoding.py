import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer, make_column_selector

hr =  pd.read_csv("HR_comma_sep.csv")

ohc =OneHotEncoder(sparse_output=False, drop = 'first')

ct = make_column_transformer((ohc, make_column_selector (dtype_include = object) ), 
                             ('passthrough', make_column_selector (dtype_exclude = object) ),
                             verbose_feature_names_out = False).set_output (transform = 'pandas')

X = hr.drop('left', axis = 1)
X = ct.fit_transform(X)
y = hr['left']

params = {'min_samples_split': np.arange(2,35,5),
          'min_samples_leaf': np.arange(1,35,5),
          'max_depth': [None,4,3,2]}

kfold = StratifiedKFold(n_splits=5, shuffle= True, random_state= 24)

dtc = DecisionTreeClassifier(random_state=24)

gcv = GridSearchCV(dtc, param_grid = params, verbose = 3, cv = kfold, scoring = 'neg_log_loss')

gcv.fit(X,y)

print(gcv.best_params_)
print(gcv.best_score_)

################################ Inferencing ##################################

tst_hr = pd.read_csv('tst_hr.csv')
dum_tst_hr = ct.transform(tst_hr)

best_model = gcv.best_estimator_
best_model.predict(dum_tst_hr)