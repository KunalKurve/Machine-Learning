import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

Concrete =  pd.read_csv("Concrete_Data.csv")

y = Concrete['Strength']
X = Concrete.drop('Strength', axis = 1)


# class sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, 
# tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

stdscaler = StandardScaler()
minmax = MinMaxScaler()
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

params = {'SCL': [stdscaler, minmax, None], 
            'SVR__C':[1,2,3],
            'SVR__epsilon':np.linspace(0.001, 5, 3)
           }


############################# RADIAL #############################################

print("Radial kernel Results: ")

svr = SVR(kernel='rbf')

pipe = Pipeline([("SCL", None),("SVR", svr)])

gsv = GridSearchCV (pipe, param_grid = params, cv = kfold, scoring = 'r2', verbose=3) #, verbose=2 shows calculation

gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# Linear #############################################

print("Linear kernel Results: ")

svr = SVR(kernel='linear',  epsilon=0.1)

pipe = Pipeline([("SCL", None),("SVR", svr)])

gsv = GridSearchCV (pipe, param_grid = params, cv = kfold, scoring = 'r2', verbose=3) #, verbose=2 shows calculation

gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# Sigmoid #############################################

print("Sigmoid kernel Results: ")

svr = SVR(kernel='sigmoid',  epsilon=0.1)

pipe = Pipeline([("SCL", None),("SVR", svr)])

gsv = GridSearchCV (pipe, param_grid = params, cv = kfold, scoring = 'r2', verbose=3) #, verbose=2 shows calculation

# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

