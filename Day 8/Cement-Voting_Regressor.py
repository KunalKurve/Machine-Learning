import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import  r2_score
from warnings import filterwarnings

filterwarnings('ignore')

conc =  pd.read_csv("Concrete_Data.csv")

y = conc['Strength']
X = conc.drop('Strength', axis=1)

lr = LinearRegression()
std_slr = StandardScaler()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=24)

################################## WITHOUT WEIGHTS #####################################
 
voting = VotingRegressor([('LR',lr), ('Ridge', ridge), ('Lasso',lasso),('TREE',dtr)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=24)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred)

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred)

lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred)

dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, y_pred)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)

print("R2 Scores for:")
print("Linear R:", r2_lr)
print("Ridge:", r2_ridge)
print("Lasso:", r2_lasso)
print("Tree:", r2_dtr)
print("Voting:", r2_voting)

################################## WITH WEIGHTS #####################################

voting = VotingRegressor([('LR',lr), ('Ridge', ridge), ('Lasso',lasso),('TREE',dtr)], 
                         weights= [r2_lr, r2_ridge, r2_lasso, r2_dtr] )

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)

print("\nWeighted Voting R2 Score:", r2_voting)

################################## GRID SEARCH CV ####################################

voting = VotingRegressor([('LR',lr), ('Ridge', ridge), ('Lasso',lasso),('TREE',dtr)])

kfold = KFold(n_splits=5, random_state=24, shuffle=True)

params = {'Ridge__alpha': np.linspace(0.001, 3, 5),
          'Lasso__alpha': np.linspace(0.001, 3, 5),
          'TREE__max_depth': [None, 3, 4, 5],
          'TREE__min_samples_leaf': [2, 5, 10],
          'TREE__min_samples_split': [1, 5, 10]}

gcv = GridSearchCV(voting, param_grid = params, cv = kfold, scoring = 'r2', n_jobs = -1)
# Verbose is used to see one progress at a time
# n_jobs: Multi processing. None: default value. -1: All cores of CPU gets activated

gcv.fit(X,y)
print("\nUsing GSCV:")
print("R2 Score:", gcv.best_score_)
print("Best Parameters:\n", gcv.best_params_)

################################## RANDOMISED SEARCH CV ####################################

voting = VotingRegressor([('LR',lr), ('Ridge', ridge), ('Lasso',lasso),('TREE',dtr)])

kfold = KFold(n_splits=5, random_state=24, shuffle=True)

params = {'Ridge__alpha': np.linspace(0.001, 3, 10),
          'Lasso__alpha': np.linspace(0.001, 3, 10),
          'TREE__max_depth': [None, 3, 4, 5],
          'TREE__min_samples_leaf': [2, 4, 5, 8, 10],
          'TREE__min_samples_split': [1, 4, 5, 8, 10]}

rcv = RandomizedSearchCV(voting, param_distributions = params, random_state=24, 
                         cv = kfold,scoring = 'r2', n_jobs = -1, n_iter = 20)

# Verbose is used to see one progress at a time
# n_jobs: Multi processing. None: default value. -1: All cores of CPU gets activated

rcv.fit(X,y)
# pd_rcv = pd.DataFrame(rcv.)
print("\nUsing RSCV:")
print("R2 Score:",rcv.best_score_)
print("Best Parameters:\n",rcv.best_params_)
 
#################################################################################

